# ga-controller/app.py

import os
import uuid
import time
import random
import json
import logging
import asyncio

from typing import List, Optional, Dict
from fastapi import FastAPI, BackgroundTasks, Response
from pydantic import BaseModel, Field
import httpx
import redis
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from azure.storage.blob import BlobServiceClient

########## LOGGING CONFIGURATION ###############################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ga-controller")

########## POD ID FOR METRICS ##################################
POD = os.getenv("POD_NAME", "unknown")

########## PROMETHEUS METRICS ##################################
gen_counter            = Counter(
    'ga_generations_total',
    'Total GA generations',
    ['pod']
)
best_fitness           = Gauge(
    'ga_best_fitness',
    'Best fitness per generation',
    ['pod']
)
mean_fitness           = Gauge(
    'ga_mean_fitness',
    'Mean fitness per generation',
    ['pod']
)
gen_duration           = Gauge(
    'ga_generation_seconds',
    'Seconds per generation',
    ['pod']
)
ga_population_size     = Gauge(
    'ga_population_size',
    'Population size used by the GA',
    ['pod']
)
ga_current_generation  = Gauge(
    'ga_current_generation',
    'Current generation index of the GA',
    ['pod']
)
ga_fitness_distribution = Histogram(
    'ga_fitness_distribution',
    'Histogram of fitness scores across the population',
    ['pod'],
    buckets=[i * 0.1 for i in range(21)]
)

########## MIGRATION CONFIG ###################################
MIGRATION_KEY = "ga:migrants"

########## REDIS CLIENT ########################################
rdb = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB",   0)),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)

########## AZURE BLOB STORAGE CLIENT ###########################
blob_service = BlobServiceClient.from_connection_string(
    os.getenv("AZURE_STORAGE_CONNECTION_STRING")
)
blob_container = os.getenv("BLOB_CONTAINER")
try:
    blob_service.create_container(blob_container)
except Exception:
    pass  # container already exists or cannot be created

########## EVALUATOR SERVICE CONFIG ############################
EVALUATOR_HOST = os.getenv("EVALUATOR_HOST", "ga-evaluator")
EVALUATOR_PORT = os.getenv("EVALUATOR_PORT", "5000")
EVALUATOR_URL  = f"http://{EVALUATOR_HOST}:{EVALUATOR_PORT}/evaluate"

########## FASTAPI APP ##########################################
app = FastAPI()

########## HEALTH & METRICS ENDPOINTS ##########################
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    # Return all metrics, with 'pod' label baked in
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

########## REQUEST / RESPONSE MODELS ############################
class RunRequest(BaseModel):
    num_tasks:         int    = Field(..., gt=0)
    num_cores:         int    = Field(..., gt=0)
    population:        int    = Field(..., gt=0)
    generations:       int    = Field(..., gt=0)
    crossover_rate:    float  = Field(..., ge=0, le=1)
    mutation_rate:     float  = Field(..., ge=0, le=1)
    migration_interval:int    = Field(..., gt=0)
    num_islands:       int    = Field(..., gt=0)
    base_energy:       float  = Field(..., gt=0)
    idle_energy:       float  = Field(..., gt=0)
    seed:              Optional[int] = None
    stagnation_limit:  Optional[int] = Field(
        None, gt=1,
        description="Stop early if no improvement over this many gens"
    )

class RunResponse(BaseModel):
    job_id: str

class ExecuteRequest(RunRequest):
    job_id: str = Field(..., description="UUID of the job to execute")

########## HELPER: HTTPX WITH RETRIES ##########################
async def eval_with_retries(
    client: httpx.AsyncClient,
    payload: dict,
    retries: int = 3,
    backoff: float = 0.5
) -> float:
    for attempt in range(1, retries + 1):
        try:
            resp = await client.post(EVALUATOR_URL, json=payload, timeout=10.0)
            resp.raise_for_status()
            return resp.json()["fitness"]
        except Exception as e:
            if attempt == retries:
                logger.error(f"Evaluator call failed after {retries} tries: {e}")
                raise
            await asyncio.sleep(backoff * attempt)

########### API ENDPOINTS ######################################
@app.post("/run", response_model=RunResponse)
async def start_run(req: RunRequest, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())
    redis_key = f"job:{job_id}"
    # initialize status in Redis
    rdb.hset(redis_key, mapping={
        "status":     "running",
        "generation": "0",
        "best":       ""
    })

    # background GA for this pod
    bg.add_task(run_ga, job_id, req.dict())

    # fan-out to all islands
    async def fan_out():
        async with httpx.AsyncClient() as client:
            tasks = []
            for i in range(req.num_islands):
                host = (
                    f"ga-controller-{i}"
                    f".ga-controller-headless.default.svc.cluster.local:8000"
                )
                payload = {"job_id": job_id, **req.dict()}
                tasks.append(
                    client.post(f"http://{host}/execute", json=payload, timeout=10.0)
                )
            await asyncio.gather(*tasks, return_exceptions=True)

    bg.add_task(fan_out)
    return RunResponse(job_id=job_id)

@app.post("/execute")
async def execute_island(req: ExecuteRequest):
    # each island runs its own GA loop
    await run_ga(req.job_id, req.dict(exclude={"job_id"}))
    return {"status": "accepted", "pod": POD}

@app.get("/run/{job_id}/status")
def status(job_id: str):
    key = f"job:{job_id}"
    if not rdb.exists(key):
        return {"error": "not found"}
    data = rdb.hgetall(key)
    return {
        "status":     data.get("status"),
        "generation": int(data.get("generation", 0)),
        "best":       float(data["best"]) if data.get("best") else None,
        "individual": json.loads(data.get("individual", "[]"))
    }

@app.get("/run/{job_id}/result")
def result(job_id: str):
    key = f"job:{job_id}"
    if not rdb.exists(key):
        return {"error": "not found"}
    status = rdb.hget(key, "status")
    if status == "running":
        return {"error": "still running"}
        
    data = rdb.hgetall(key)
    response = {
        "status":        status,
        "generation":    int(data.get("generation", 0)),
        "best_fitness":  float(data.get("best", 0.0)) if data.get("best") else None,
        "individual":    json.loads(data.get("individual", "[]"))
    }

if status == "done":
        blob_client = blob_service.get_blob_client(
            container=blob_container,
            blob=f"{job_id}.json"
        )
        blob_data   = blob_client.download_blob().readall()
        full_result = json.loads(blob_data)
        response.update(full_result)

    return response
########## GA UTILITIES #######################################
def init_population(pop_size: int, num_tasks: int,
                    num_cores: int, rng: random.Random
                   ) -> List[List[int]]:
    return [
        [rng.randint(0, num_cores - 1) for _ in range(num_tasks)]
        for _ in range(pop_size)
    ]

def next_generation(population: List[List[int]],
                    fitnesses:   List[float],
                    cfg:         Dict) -> List[List[int]]:
    pop_size = len(population)
    tour_size = 3

    # Tournament selection
    selected = []
    for _ in range(pop_size):
        aspirants = random.sample(list(zip(population, fitnesses)), tour_size)
        winner    = min(aspirants, key=lambda x: x[1])[0]
        selected.append(winner.copy())

    # Crossover
    offspring = []
    for i in range(0, pop_size, 2):
        p1, p2 = selected[i], selected[(i+1) % pop_size]
        if random.random() < cfg['crossover_rate']:
            pt = random.randint(1, len(p1)-1)
            offspring += [p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]]
        else:
            offspring += [p1.copy(), p2.copy()]

    # Mutation
    for ind in offspring:
        if random.random() < cfg['mutation_rate']:
            idx = random.randrange(len(ind))
            ind[idx] = random.randint(0, cfg['num_cores']-1)

    return offspring[:pop_size]

def compute_core_times(individual: List[int],
                       exec_times:  List[float],
                       num_cores:   int) -> List[float]:
    cores = [0.0]*num_cores
    for i, c in enumerate(individual):
        cores[c] += exec_times[i]
    return cores

########## MAIN GA LOOP #######################################
async def run_ga(job_id: str, cfg: Dict):
    key = f"job:{job_id}"
    stopped_due_to_stagnation = False

    # Pull parameters
    interval       = cfg["migration_interval"]
    num_islands    = cfg["num_islands"]
    stagnation_lim = cfg.get("stagnation_limit") or 0

    prev_best       = float('inf')
    best_individual = None
    no_improve      = 0

    # Generate execution times
    base_seed  = cfg.get("seed") or 0
    rng_exec   = random.Random(base_seed)
    exec_times = [rng_exec.randint(1,10) for _ in range(cfg["num_tasks"])]
    logger.info(f"Job {job_id}: exec_times (first 5): {exec_times[:5]}")

    # Island‐specific RNG
    pod_name  = os.getenv("POD_NAME", "ga-island-0")
    island_id = int(pod_name.rsplit("-",1)[-1]) if "-" in pod_name else 0
    rng_pop   = random.Random(base_seed + island_id)

    # Initialize population
    population = init_population(
        cfg["population"], cfg["num_tasks"], cfg["num_cores"], rng_pop
    )
    ga_population_size.labels(pod=POD).set(len(population))
    
    try:
        for gen in range(1, cfg["generations"] + 1):
            start = time.time()

            # parallel fitness evaluation
            async with httpx.AsyncClient() as client:
                tasks     = [
                    eval_with_retries(client, {
                        "individual":      indiv,
                        "execution_times": exec_times,
                        "base_energy":     cfg["base_energy"],
                        "idle_energy":     cfg["idle_energy"],
                    })
                    for indiv in population
                ]
                fitnesses = await asyncio.gather(*tasks)

            best     = min(fitnesses)
            mean_val = sum(fitnesses) / len(fitnesses)
    
            # record metrics with pod label
            gen_counter.labels(pod=POD).inc()
            ga_current_generation.labels(pod=POD).set(gen)
            best_fitness.labels(pod=POD).set(best)
            mean_fitness.labels(pod=POD).set(mean_val)
            gen_duration.labels(pod=POD).set(time.time() - start)
            for f in fitnesses:
                ga_fitness_distribution.labels(pod=POD).observe(f)

            logger.info(f"Job {job_id} Gen {gen}: best={best:.4f}, mean={mean_val:.4f}")

            # update Redis with the previous best (so status endpoint can show progress)
            rdb.hset(key, mapping={
                "generation": str(gen),
                "best":       str(prev_best if prev_best < float('inf') else best),
                "individual": json.dumps(best_individual) if best_individual else ""
            })
    
            # track global best & stagnation
            if best < prev_best:
                idx            = fitnesses.index(best)
                best_individual = population[idx]
                prev_best       = best
                no_improve      = 0

                # publish new global best for migration
                rdb.hset(key, mapping={
                    "generation": str(gen),
                    "best":       str(best),
                    "individual": json.dumps(best_individual)
                })
                rdb.lpush(MIGRATION_KEY, json.dumps(best_individual))
                rdb.ltrim(MIGRATION_KEY, 0, num_islands - 1)
            else:
                if stagnation_lim > 0:
                    no_improve += 1
                    if no_improve >= stagnation_lim:
                        stopped_due_to_stagnation = True
                        logger.info(
                            f"Job {job_id}: no improvement for {no_improve} gens "
                            f"(limit={stagnation_lim}), stopping early at gen={gen}"
                        )
                        break
    
            # migration step
            if gen % interval == 0:
                migrants_raw = rdb.lrange(MIGRATION_KEY, 0, num_islands - 1)
                migrants     = [json.loads(m) for m in migrants_raw]
                async with httpx.AsyncClient() as client:
                    tasks        = [
                        eval_with_retries(client, {
                            "individual":      m,
                            "execution_times": exec_times,
                            "base_energy":     cfg["base_energy"],
                            "idle_energy":     cfg["idle_energy"]
                        })
                        for m in migrants
                    ]
                    migrant_fits = await asyncio.gather(*tasks)
                pairs = list(zip(population, fitnesses))
                pairs.sort(key=lambda x: x[1], reverse=True)
                for i, fit in enumerate(migrant_fits):
                    pairs[i] = (migrants[i], fit)
                population = [ind for ind,_ in pairs]
                rdb.delete(MIGRATION_KEY)
    
            # next gen
            population = next_generation(population, fitnesses, cfg)
    
        # build final result (include best individual)
        final = {
            "job_id":           job_id,
            "best_fitness":     prev_best,
            "best_individual":  best_individual,
            "core_times":       compute_core_times(best_individual, exec_times, cfg["num_cores"]),
            "GA_runtime": time.time() - start, 
            "metrics": {
                "generations_executed": gen,
                "total_duration_s":     sum(sample.value for sample in gen_duration.collect())
            }
        }
        blob_client = blob_service.get_blob_client(container=blob_container, blob=f"{job_id}.json")
        blob_client.upload_blob(json.dumps(final), overwrite=True)
        logger.info(f"Job {job_id}: results uploaded to Blob")
    except Exception as e:
        # catches ANY exception in the GA loop or upload
        logger.exception(f"Job {job_id} failed: {e}")
        rdb.hset(key, "status", "failed")
    else:
        # only if no exception at all
        rdb.hset(key, "status", "done")
    finally:
        # clean up the migrant list
        rdb.delete(MIGRATION_KEY)
        logger.info(f"Job {job_id}: cleanup complete")
