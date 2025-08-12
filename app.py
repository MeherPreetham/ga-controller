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

from typing import List, Optional, Dict, Tuple
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
    stagnation_limit:   Optional[int] = Field(None, ge=0, description="0 disables early stop; otherwise stop after N gens w/o global improvement")
    case_label:         Optional[str] = None

class RunResponse(BaseModel):
    job_id: str

class ExecuteRequest(RunRequest):
    job_id: str = Field(..., description="Job to execute on this island")

########## HELPERS #############################################
def migrants_key(job_id: str) -> str:
    return f"ga:{job_id}:migrants"

def is_leader(job_id: str) -> bool:
    return rdb.get(f"job:{job_id}:leader") == POD

def get_stop_flag(job_id: str) -> bool:
    return rdb.get(f"job:{job_id}:stop") == "1"

def set_stop_flag(job_id: str):
    rdb.set(f"job:{job_id}:stop", "1")

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

########## GLOBAL-BEST UPDATE ###########################
def try_update_global_best(job_id: str, cand_best: float, cand_individual: List[int], gen: int) -> Tuple[bool, float]:
    """
    Atomically update Redis global best if cand_best is strictly better.
    Also updates last_improve_gen.
    Returns (updated, new_global_best).
    """
    key = f"job:{job_id}"
    pipe = rdb.pipeline()
    while True:
        try:
            pipe.watch(key)
            curr_str = pipe.hget(key, "best")
            curr = float(curr_str) if curr_str else float("inf")
            if cand_best >= curr:
                pipe.unwatch()
                return False, curr
            # update global best, individual, and last improvement generation atomically
            pipe.multi()
            pipe.hset(key, mapping={
                "best":               str(cand_best),
                "individual":         json.dumps(cand_individual),
                "last_improve_gen":   str(gen)
            })
            pipe.execute()
            return True, cand_best
        except redis.WatchError:
            continue
        finally:
            try:
                pipe.reset()
            except Exception:
                pass

########## API: START RUN ######################################
@app.post("/run", response_model=RunResponse)
async def start_run(req: RunRequest):
    if not AZ_CONN_STR or not AZ_CONTAINER:
        raise HTTPException(status_code=500, detail="Missing AZURE_STORAGE_CONNECTION_STRING or BLOB_CONTAINER")

    job_id    = str(uuid.uuid4())
    redis_key = f"job:{job_id}"

    # Elect a leader (first pod to create job)
    rdb.setnx(f"{redis_key}:leader", POD)

    # initialize job keys
    rdb.hset(redis_key, mapping={
        "status": "running",
        "generation": "0",
        "best": "",                # global best
        "individual": "",          # global best individual
        "last_improve_gen": "0"
    })
    rdb.set(f"{redis_key}:stop", "0")  # global stop flag
    rdb.delete(migrants_key(job_id))   # clear job-scoped migrant list

    cfg = req.dict()

    # 1) run GA on this pod
    asyncio.create_task(run_ga(job_id, cfg))
    # 2) fan-out to islands
    asyncio.create_task(fan_out(job_id, cfg))

    return RunResponse(job_id=job_id)

@app.post("/execute")
async def execute_island(req: ExecuteRequest):
    await run_ga(req.job_id, req.dict(exclude={"job_id"}))
    return {"status": "accepted", "pod": POD}

########## API: STATUS ######################
@app.get("/status/{job_id}")
def status(job_id: str):
    key = f"job:{job_id}"
    if not rdb.exists(key):
        raise HTTPException(404, "Job not found")
    data = rdb.hgetall(key)
    return {
        "status":          data.get("status"),
        "generation":      int(data.get("generation", 0)),
        "best_fitness":    float(data["best"]) if data.get("best") else None,
        "best_individual": json.loads(data.get("individual", "[]")) if data.get("individual") else None
    }

########## API: RESULT ######################
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

########## MAIN GA LOOP (island) ###############################
async def run_ga(job_id: str, cfg: Dict):
    key = f"job:{job_id}"
    mkey = migrants_key(job_id)
    stagnated_local = False  # island-local stagnation

    # Azure blob client
    svc = BlobServiceClient.from_connection_string(AZ_CONN_STR)
    try:
        svc.create_container(AZ_CONTAINER)
    except Exception:
        pass

    # Prepare execution times (seeded)
    base_seed = int(cfg.get("seed") or 0)
    rng_exec  = random.Random(base_seed)
    exec_times = [rng_exec.randint(10, 20) for _ in range(cfg["num_tasks"])]
    logger.info(f"Job {job_id} ({POD}): exec_times head={exec_times[:5]}")

    # Island RNG (so each island differs deterministically)
    pod_name  = os.getenv("POD_NAME", "ga-island-0")
    island_id = int(pod_name.rsplit("-", 1)[-1]) if "-" in pod_name and pod_name.rsplit("-", 1)[-1].isdigit() else 0
    rng_pop   = random.Random(base_seed + island_id)

    # Initialize population
    population = init_population(cfg["population"], cfg["num_tasks"], cfg["num_cores"], rng_pop)

    # Per-generation stats (local, for plotting)
    best_per_gen: List[float] = []
    avg_per_gen:  List[float] = []
    std_per_gen:  List[float] = []

    # Island-local best (for migration decisions)
    local_best: float = float('inf')
    local_best_ind: Optional[List[int]] = None

    no_improve_local = 0
    stagnation_lim = int(cfg.get("stagnation_limit") or 0)
    interval = cfg["migration_interval"]
    num_islands = cfg["num_islands"]

    start_all = time.time()
    gen_executed = 0

    # HTTP client + concurrency limits
    MAX_CONC = int(os.getenv("EVAL_CONCURRENCY", "24"))
    limits = httpx.Limits(max_connections=64, max_keepalive_connections=32)

    try:
        for gen in range(1, cfg["generations"] + 1):
            # Early stop if leader set global stop
            if get_stop_flag(job_id):
                logger.info(f"Job {job_id} ({POD}): observed global stop at gen={gen}")
                break

            sem = asyncio.Semaphore(MAX_CONC)
            async with httpx.AsyncClient(timeout=30.0, limits=limits) as client:
                async def safe_eval(indiv):
                    async with sem:
                        try:
                            return await eval_with_retries(client, {
                                "individual":      indiv,
                                "execution_times": exec_times,
                                "base_energy":     cfg["base_energy"],
                                "idle_energy":     cfg["idle_energy"],
                            })
                        except Exception as e:
                            logger.error(f"Job {job_id} ({POD}) gen={gen}: evaluator failed: {e}")
                            return float("inf")

                # Evaluate population
                tasks = [safe_eval(indiv) for indiv in population]
                fitnesses = await asyncio.gather(*tasks)

            # Per-gen stats
            best = min(fitnesses)
            avgv = mean(fitnesses)
            stdv = pstdev(fitnesses) if len(fitnesses) > 1 else 0.0

            best_per_gen.append(best)
            avg_per_gen.append(avgv)
            std_per_gen.append(stdv)
            gen_executed = gen

            # Progress: update generation only (do NOT overwrite global best here)
            rdb.hset(key, mapping={"generation": str(gen)})

            # If island improved locally, attempt atomic GLOBAL best update
            if best < local_best:
                idx            = fitnesses.index(best)
                local_best_ind = population[idx]
                local_best     = best

                # Attempt atomic global-best update (also records last_improve_gen)
                try_update_global_best(job_id, local_best, local_best_ind, gen)
                # Reset local stagnation on local improvement (not only when global improves)
                no_improve_local = 0
            else:
                # Local stagnation handling (optional)
                if stagnation_lim > 0:
                    no_improve_local += 1
                    if no_improve_local >= stagnation_lim:
                        stagnated_local = True
                        logger.info(f"Job {job_id} ({POD}): local stagnation at gen={gen} (limit={stagnation_lim})")
                        # Do NOT set job status; leader decides global stop
                        break

            # Leader-only: decide global early stop based on GLOBAL last_improve_gen
            if stagnation_lim > 0 and is_leader(job_id):
                try:
                    last_improve = int(rdb.hget(key, "last_improve_gen") or "0")
                except Exception:
                    last_improve = 0
                if gen - last_improve >= stagnation_lim:
                    logger.info(f"Job {job_id} (leader {POD}): global stagnation detected at gen={gen} (last_improve_gen={last_improve}, limit={stagnation_lim})")
                    set_stop_flag(job_id)

            # Migration USING CURRENT MIGRANTS LIST
                migrants_raw = rdb.lrange(mkey, 0, num_islands - 1)
                migrants     = [json.loads(m) for m in migrants_raw] if migrants_raw else []
                if migrants:
                    sem_m = asyncio.Semaphore(MAX_CONC)
                    async with httpx.AsyncClient(timeout=30.0, limits=limits) as client:
                        async def safe_eval_mig(m):
                            async with sem_m:
                                try:
                                    return await eval_with_retries(client, {
                                        "individual":      m,
                                        "execution_times": exec_times,
                                        "base_energy":     cfg["base_energy"],
                                        "idle_energy":     cfg["idle_energy"]
                                    })
                                except Exception as e:
                                    logger.error(f"Job {job_id} ({POD}) gen={gen}: migrant eval failed: {e}")
                                    return float("inf")

                        tasks = [safe_eval_mig(m) for m in migrants]
                        migrant_fits = await asyncio.gather(*tasks)

                    # Replace worst with migrants
                    pairs = list(zip(population, fitnesses))
                    pairs.sort(key=lambda x: x[1], reverse=True)
                    for i, fit in enumerate(migrant_fits):
                        if i < len(pairs):
                            pairs[i] = (migrants[i], fit)
                    population = [ind for ind, _ in pairs]
                    rdb.delete(mkey)

            # Next generation
            population = next_generation(population, fitnesses, cfg)

    except Exception as e:
        logger.exception(f"Job {job_id} ({POD}): fatal GA error: {e}")
        if is_leader(job_id):
            rdb.hset(key, mapping={"status": "error", "generation": str(gen_executed)})
        return

    elapsed_all = time.time() - start_all

    # Leader compiles final output & uploads blob; others just exit
    if is_leader(job_id):
        gbest_str = rdb.hget(key, "best")
        gbest = float(gbest_str) if gbest_str else (best_per_gen[-1] if best_per_gen else float("inf"))
        gind_json = rdb.hget(key, "individual")
        gind = json.loads(gind_json) if gind_json else (local_best_ind or (population[0] if population else []))

        cores, mk, te, imb = compute_raw_metrics(gind, exec_times, cfg["num_cores"],
                                                 cfg["base_energy"], cfg["idle_energy"])

        final = {
            "run_id":               job_id,
            "case_label":           cfg.get("case_label"),
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
            "generations_executed": len(best_per_gen),
            "elapsed_time_s":       elapsed_all,
            "best_individual":      gind,
            "best_fitness":         gbest,
            "best_per_generation":  best_per_gen,
            "avg_per_generation":   avg_per_gen,
            "std_per_generation":   std_per_gen,
            "makespan":             mk,
            "total_energy":         te,
            "imbalance":            imb,
            "core_times":           cores
        }

        try:
            txt_blob = svc.get_blob_client(container=AZ_CONTAINER, blob=f"{job_id}.txt")
            txt_blob.upload_blob(json.dumps(final), overwrite=True)

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            buf = io.BytesIO()
            plt.figure()
            plt.plot(final["best_per_generation"], label="Best")
            plt.plot(final["avg_per_generation"],  label="Average")
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            plt.legend()
            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=150)
            plt.close()
            buf.seek(0)
            png_blob = svc.get_blob_client(container=AZ_CONTAINER, blob=f"{job_id}.png")
            png_blob.upload_blob(buf.read(), overwrite=True)
        except Exception as e:
            logger.error(f"Blob upload failed for job {job_id}: {e}")

        stopped_globally = get_stop_flag(job_id)
        final_status = "stagnated" if stopped_globally else ("done" if not stagnated_local else "stagnated")

        rdb.hset(key, mapping={"status": final_status, "generation": str(gen_executed)})
        rdb.delete(migrants_key(job_id))
        rdb.delete(f"job:{job_id}:stop")
        logger.info(f"Job {job_id}: finalized by leader {POD} → status={final_status}")
    else:
        logger.info(f"Job {job_id}: island {POD} finished (leader will finalize)")
