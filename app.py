# ga-controller/app.py
import os
import uuid
import time
import json
import logging
import asyncio
import socket
import random
import io

from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx
import redis
import matplotlib.pyplot as plt
from statistics import mean, pstdev
from azure.storage.blob import BlobServiceClient

########## LOGGING #############################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ga-controller")

########## POD & DISCOVERY #####################################
POD = os.getenv("POD_NAME", "unknown")
CONTROLLER_HEADLESS = os.getenv("CONTROLLER_HEADLESS", "ga-controller-headless.default.svc.cluster.local")
CONTROLLER_PORT     = int(os.getenv("CONTROLLER_PORT", "8000"))

########## REDIS ###############################################
rdb = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB",   0)),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)

########## AZURE BLOB (created lazily) #########################
AZ_CONN_STR   = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZ_CONTAINER  = os.getenv("BLOB_CONTAINER")

########## EVALUATOR ###########################################
EVALUATOR_HOST = os.getenv("EVALUATOR_HOST", "ga-evaluator")
EVALUATOR_PORT = os.getenv("EVALUATOR_PORT", "5000")
EVALUATOR_URL  = f"http://{EVALUATOR_HOST}:{EVALUATOR_PORT}/evaluate"

########## FASTAPI #############################################
app = FastAPI(title="GA Controller (islands)")

########## HEALTH ##############################################
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

########## MODELS ##############################################
class RunRequest(BaseModel):
    # GA config (keep existing names for controller, add optional case_label)
    num_tasks:          int   = Field(..., gt=0)
    num_cores:          int   = Field(..., gt=0)
    population:         int   = Field(..., gt=1)
    generations:        int   = Field(..., gt=0)
    crossover_rate:     float = Field(..., ge=0, le=1)
    mutation_rate:      float = Field(..., ge=0, le=1)
    migration_interval: int   = Field(..., gt=0)
    num_islands:        int   = Field(..., gt=0)
    base_energy:        float = Field(..., gt=0)
    idle_energy:        float = Field(..., ge=0)
    seed:               Optional[int] = None
    stagnation_limit:   Optional[int] = Field(None, gt=1)
    case_label:         Optional[str] = None

class RunResponse(BaseModel):
    job_id: str

class ExecuteRequest(RunRequest):
    job_id: str = Field(..., description="Job to execute on this island")

########## EVALUATOR HELPER ####################################
async def eval_with_retries(
    client: httpx.AsyncClient,
    payload: dict,
    retries: int = 3,
    backoff: float = 0.5
) -> float:
    for attempt in range(1, retries + 1):
        try:
            resp = await client.post(EVALUATOR_URL, json=payload, timeout=20.0)
            resp.raise_for_status()
            data = resp.json()
            return float(data["fitness"])
        except Exception as e:
            if attempt == retries:
                logger.error(f"Evaluator call failed after {retries} tries: {e}")
                raise
            await asyncio.sleep(backoff * attempt)

########## ISLAND FAN-OUT ######################################
async def fan_out(job_id: str, cfg: Dict):
    try:
        infos = socket.getaddrinfo(CONTROLLER_HEADLESS, CONTROLLER_PORT, proto=socket.IPPROTO_TCP)
        hosts = sorted({f"{addr[4][0]}:{CONTROLLER_PORT}" for addr in infos})
    except Exception as e:
        logger.warning(f"Headless discovery failed; running locally only. {e}")
        hosts = []

    if not hosts:
        return

    logger.info(f"Dispatching Job {job_id} to islands: {hosts}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [
            client.post(f"http://{host}/execute", json={"job_id": job_id, **cfg})
            for host in hosts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for host, res in zip(hosts, results):
            if isinstance(res, Exception):
                logger.error(f"Island {host} fan-out FAILED: {res}")
            else:
                logger.info(f"Island {host} fan-out OK: {res.status_code}")

########## SIMPLE GA UTILS #####################################
def init_population(pop_size: int, num_tasks: int, num_cores: int, rng: random.Random) -> List[List[int]]:
    return [[rng.randint(0, num_cores - 1) for _ in range(num_tasks)] for _ in range(pop_size)]

def next_generation(population: List[List[int]], fitnesses: List[float], cfg: Dict) -> List[List[int]]:
    pop_size = len(population)
    tour_size = 3

    # Tournament selection
    selected = []
    for _ in range(pop_size):
        aspirants = random.sample(list(zip(population, fitnesses)), tour_size)
        winner    = min(aspirants, key=lambda x: x[1])[0]
        selected.append(winner.copy())

    # One-point crossover
    offspring = []
    for i in range(0, pop_size, 2):
        p1, p2 = selected[i], selected[(i+1) % pop_size]
        if random.random() < cfg['crossover_rate']:
            pt = random.randint(1, len(p1) - 1)
            offspring += [p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]]
        else:
            offspring += [p1.copy(), p2.copy()]

    # Mutation (single gene)
    for ind in offspring:
        if random.random() < cfg['mutation_rate']:
            idx = random.randrange(len(ind))
            ind[idx] = random.randint(0, cfg['num_cores'] - 1)

    return offspring[:pop_size]

def compute_core_times(individual: List[int], exec_times: List[float], num_cores: int) -> List[float]:
    cores = [0.0] * num_cores
    for i, c in enumerate(individual):
        cores[c] += exec_times[i]
    return cores

def compute_raw_metrics(individual: List[int], exec_times: List[float], num_cores: int,
                        base_energy: float, idle_energy: float):
    cores = compute_core_times(individual, exec_times, num_cores)
    makespan = max(cores) if cores else 0.0
    active_e = sum(ct * base_energy for ct in cores)
    idle_e   = sum((makespan - ct) * idle_energy for ct in cores)
    total_e  = active_e + idle_e
    mean_load = (sum(exec_times) / max(num_cores, 1)) if num_cores > 0 else 1e-9
    imbalance = pstdev(cores) / (mean_load if mean_load != 0 else 1e-9)
    return cores, makespan, total_e, imbalance

########## API: START RUN ######################################
@app.post("/run", response_model=RunResponse)
async def start_run(req: RunRequest):
    if not AZ_CONN_STR or not AZ_CONTAINER:
        raise HTTPException(status_code=500, detail="Missing AZURE_STORAGE_CONNECTION_STRING or BLOB_CONTAINER")

    job_id = str(uuid.uuid4())
    redis_key = f"job:{job_id}"

    # initialize basic status in Redis
    rdb.hset(redis_key, mapping={
        "status": "running",
        "generation": "0",
        "best": "",
        "individual": ""
    })

    cfg = req.dict()
    # 1) run GA on this pod
    asyncio.create_task(run_ga(job_id, cfg))
    # 2) fan-out to islands
    asyncio.create_task(fan_out(job_id, cfg))

    return RunResponse(job_id=job_id)

########## BACK-COMPAT (island entry) ##########################
@app.post("/execute")
async def execute_island(req: ExecuteRequest):
    await run_ga(req.job_id, req.dict(exclude={"job_id"}))
    return {"status": "accepted", "pod": POD}

########## API: STATUS (monolithic-style) ######################
@app.get("/status/{job_id}")
def status(job_id: str):
    key = f"job:{job_id}"
    if not rdb.exists(key):
        raise HTTPException(404, "Job not found")
    data = rdb.hgetall(key)
    return {
        "status":        data.get("status"),
        "generation":    int(data.get("generation", 0)),
        "best_fitness":  float(data["best"]) if data.get("best") else None,
        "best_individual": json.loads(data.get("individual", "[]")) if data.get("individual") else None
    }

# keep old paths working
@app.get("/run/{job_id}/status")
def status_compat(job_id: str):
    return status(job_id)

########## API: RESULT (monolithic-style) ######################
@app.get("/result/{job_id}")
def result(job_id: str):
    key = f"job:{job_id}"
    if not rdb.exists(key):
        raise HTTPException(404, "Job not found")

    if rdb.hget(key, "status") == "running":
        raise HTTPException(400, "Job still running")

    # Load final JSON from blob (single source of truth)
    try:
        svc  = BlobServiceClient.from_connection_string(AZ_CONN_STR)
        blob = svc.get_blob_client(container=AZ_CONTAINER, blob=f"{job_id}.txt")
        payload = json.loads(blob.download_blob().readall())
        return payload
    except Exception as e:
        logger.error(f"Failed to read final result from blob: {e}")
        raise HTTPException(500, "Failed to read final result from blob")

# keep old path working
@app.get("/run/{job_id}/result")
def result_compat(job_id: str):
    return result(job_id)

########## MAIN GA LOOP (island) ###############################
MIGRATION_KEY = "ga:migrants"

async def run_ga(job_id: str, cfg: Dict):
    key = f"job:{job_id}"
    stagnated = False

    # Azure blob client (lazy)
    svc = BlobServiceClient.from_connection_string(AZ_CONN_STR)
    try:
        svc.create_container(AZ_CONTAINER)
    except Exception:
        pass

    # Prepare execution times (seeded)
    base_seed = int(cfg.get("seed") or 0)
    rng_exec  = random.Random(base_seed)
    exec_times = [rng_exec.randint(10, 20) for _ in range(cfg["num_tasks"])]
    logger.info(f"Job {job_id}: exec_times head={exec_times[:5]}")

    # Island RNG (so each island differs deterministically)
    pod_name  = os.getenv("POD_NAME", "ga-island-0")
    island_id = int(pod_name.rsplit("-", 1)[-1]) if "-" in pod_name and pod_name.rsplit("-", 1)[-1].isdigit() else 0
    rng_pop   = random.Random(base_seed + island_id)

    # Initialize population
    population = init_population(cfg["population"], cfg["num_tasks"], cfg["num_cores"], rng_pop)

    # Per-generation stats
    best_per_gen: List[float] = []
    avg_per_gen:  List[float] = []
    std_per_gen:  List[float] = []

    prev_best: float = float('inf')
    best_individual: Optional[List[int]] = None
    no_improve = 0
    stagnation_lim = int(cfg.get("stagnation_limit") or 0)
    interval = cfg["migration_interval"]
    num_islands = cfg["num_islands"]

    start_all = time.time()
    gen_executed = 0

    for gen in range(1, cfg["generations"] + 1):
        # Evaluate population in parallel via evaluator service
        async with httpx.AsyncClient() as client:
            tasks = [
                eval_with_retries(client, {
                    "individual":      indiv,
                    "execution_times": exec_times,
                    "base_energy":     cfg["base_energy"],
                    "idle_energy":     cfg["idle_energy"],
                })
                for indiv in population
            ]
            fitnesses = await asyncio.gather(*tasks)

        # Per-gen stats
        best = min(fitnesses)
        avgv = mean(fitnesses)
        stdv = pstdev(fitnesses) if len(fitnesses) > 1 else 0.0

        best_per_gen.append(best)
        avg_per_gen.append(avgv)
        std_per_gen.append(stdv)
        gen_executed = gen

        # Update Redis progress with *current* best (even if not global-best)
        rdb.hset(key, mapping={
            "generation": str(gen),
            "best":       str(best if prev_best == float('inf') else min(prev_best, best)),
            "individual": json.dumps(best_individual or [])
        })

        # Update global best + stagnation
        if best < prev_best:
            idx             = fitnesses.index(best)
            best_individual = population[idx]
            prev_best       = best
            no_improve      = 0
            rdb.hset(key, mapping={
                "best":       str(prev_best),
                "individual": json.dumps(best_individual)
            })
            # publish migrant
            rdb.lpush(MIGRATION_KEY, json.dumps(best_individual))
            rdb.ltrim(MIGRATION_KEY, 0, num_islands - 1)
        else:
            if stagnation_lim > 0:
                no_improve += 1
                if no_improve >= stagnation_lim:
                    stagnated = True
                    logger.info(f"Job {job_id}: stagnated at gen={gen}")
                    break

        # Migration
        if gen % interval == 0:
            migrants_raw = rdb.lrange(MIGRATION_KEY, 0, num_islands - 1)
            migrants     = [json.loads(m) for m in migrants_raw] if migrants_raw else []
            if migrants:
                async with httpx.AsyncClient() as client:
                    tasks = [
                        eval_with_retries(client, {
                            "individual":      m,
                            "execution_times": exec_times,
                            "base_energy":     cfg["base_energy"],
                            "idle_energy":     cfg["idle_energy"]
                        })
                        for m in migrants
                    ]
                    migrant_fits = await asyncio.gather(*tasks)

                # Replace worst with migrants
                pairs = list(zip(population, fitnesses))
                pairs.sort(key=lambda x: x[1], reverse=True)
                for i, fit in enumerate(migrant_fits):
                    if i < len(pairs):
                        pairs[i] = (migrants[i], fit)
                population = [ind for ind, _ in pairs]
                rdb.delete(MIGRATION_KEY)

        # Next generation
        population = next_generation(population, fitnesses, cfg)

    # Ensure we have a best individual
    if best_individual is None:
        # fall back to the first individual if nothing improved (edge case)
        best_individual = population[0]
        cores, mk, te, imb = compute_raw_metrics(best_individual, exec_times, cfg["num_cores"],
                                                 cfg["base_energy"], cfg["idle_energy"])
        prev_best = best_per_gen[-1] if best_per_gen else float('inf')
    else:
        cores, mk, te, imb = compute_raw_metrics(best_individual, exec_times, cfg["num_cores"],
                                                 cfg["base_energy"], cfg["idle_energy"])

    elapsed_all = time.time() - start_all

    # Build final JSON in the SAME SHAPE as monolithic GA
    final = {
        # identity / labeling
        "run_id":               job_id,
        "case_label":           cfg.get("case_label"),

        # config snapshot (use monolithic key names for parity)
        "num_tasks":            cfg["num_tasks"],
        "num_cores":            cfg["num_cores"],
        "num_population":       cfg["population"],
        "num_generations":      cfg["generations"],
        "crossover_rate":       cfg["crossover_rate"],
        "mutation_rate":        cfg["mutation_rate"],
        "base_energy":          cfg["base_energy"],
        "idle_energy":          cfg["idle_energy"],
        "stagnation_limit":     cfg.get("stagnation_limit"),
        "seed":                 cfg.get("seed"),

        # outcomes
        "generations_executed": len(best_per_gen),
        "elapsed_time_s":       elapsed_all,
        "best_individual":      best_individual,
        "best_fitness":         prev_best,
        "best_per_generation":  best_per_gen,
        "avg_per_generation":   avg_per_gen,
        "std_per_generation":   std_per_gen,

        # raw metrics of final best
        "makespan":             mk,
        "total_energy":         te,
        "imbalance":            imb,
        "core_times":           cores
    }

    # Upload JSON + plot to blob
    try:
        txt_blob = svc.get_blob_client(container=AZ_CONTAINER, blob=f"{job_id}.txt")
        txt_blob.upload_blob(json.dumps(final), overwrite=True)

        # Plot and upload PNG
        plt.figure()
        plt.plot(final["best_per_generation"], label="Best")
        plt.plot(final["avg_per_generation"],  label="Average")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close()
        buf.seek(0)
        png_blob = svc.get_blob_client(container=AZ_CONTAINER, blob=f"{job_id}.png")
        png_blob.upload_blob(buf.read(), overwrite=True)
    except Exception as e:
        logger.error(f"Blob upload failed for job {job_id}: {e}")

    # Mark done in Redis
    rdb.hset(key, mapping={
        "status": "stagnated" if stagnated else "done",
        "generation": str(gen_executed),
        "best": str(prev_best),
        "individual": json.dumps(best_individual or [])
    })
    rdb.delete(MIGRATION_KEY)
    logger.info(f"Job {job_id}: complete → status={'stagnated' if stagnated else 'done'}")
