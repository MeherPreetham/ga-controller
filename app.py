# ga-controller/app.py

import os
import uuid
import time
import random
import json
import logging

from typing import List, Optional, Dict
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
import httpx
import redis
from prometheus_client import Counter, Gauge, start_http_server
from azure.storage.blob import BlobServiceClient

########## LOGGING CONFIGURATION ###############################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ga-controller")

############## PROMETHEUS METRICS ##################################
gen_counter  = Counter('ga_generations_total', 'Total GA generations')
best_fitness = Gauge('ga_best_fitness',    'Best fitness per generation')
mean_fitness = Gauge('ga_mean_fitness',    'Mean fitness per generation')
gen_duration = Gauge('ga_generation_seconds', 'Seconds per generation')

################ REDIS CLIENT ######################################
rdb = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0)),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)
################ AZURE BLOB STORAGE CLIENT ########################
blob_service = BlobServiceClient.from_connection_string(
    os.getenv("AZURE_STORAGE_CONNECTION_STRING")
)
blob_container = os.getenv("BLOB_CONTAINER")
try:
    blob_service.create_container(blob_container)
except Exception:
    pass

################ EVALUATOR SERVICE CONFIG ########################
EVALUATOR_HOST = os.getenv("EVALUATOR_HOST", "ga-evaluator")
EVALUATOR_PORT = os.getenv("EVALUATOR_PORT", "5000")
EVALUATOR_URL  = f"http://{EVALUATOR_HOST}:{EVALUATOR_PORT}/evaluate"


################ FASTAPI APP ######################################
app = FastAPI()
start_http_server(8000)

class RunRequest(BaseModel):
    num_tasks:        int   = Field(..., gt=0, description="Number of tasks")
    num_cores:        int   = Field(..., gt=0, description="Number of cores")
    population:       int   = Field(..., gt=0, description="GA population size")
    generations:      int   = Field(..., gt=0, description="Number of generations")
    crossover_rate:   float = Field(..., ge=0, le=1, description="Crossover probability")
    mutation_rate:    float = Field(..., ge=0, le=1, description="Mutation probability")
    base_energy:      float = Field(..., gt=0, description="Energy per active unit time")
    idle_energy:      float = Field(..., gt=0, description="Energy per idle unit time")
    seed:             Optional[int] = Field(None, description="Optional RNG seed")

class RunResponse(BaseModel):
    job_id: str

########### API ENDPOINTS ################################################
@app.post("/run", response_model=RunResponse)
def start_run(req: RunRequest, bg: BackgroundTasks):
    params = req.dict()
    job_id = str(uuid.uuid4())
    redis_key = f"job:{job_id}"
    rdb.hset(redis_key, mapping={"status": "running", "generation": "0", "best": ""})
    bg.add_task(run_ga, job_id, params)
    return RunResponse(job_id=job_id)

@app.get("/run/{job_id}/status")
def status(job_id: str):
    logger.debug(f"Status requested for job_id={job_id}")
    key = f"job:{job_id}"
    if not rdb.exists(key):
        logger.warning(f"Status for unknown job_id={job_id}")
        return {"error": "not found"}
    data = rdb.hgetall(key)
    return {
      "status":     data.get("status"),
      "generation": int(data.get("generation", 0)),
      "best":       float(data["best"]) if data.get("best") else None
    }
@app.get("/run/{job_id}/result")
def result(job_id: str):
    logger.debug(f"Result requested for job_id={job_id}")
    key = f"job:{job_id}"
    if rdb.exists(key):
        data = rdb.hgetall(key)
        if data.get("status") != "done":
            return {"error": "still running"}
        
    # If Redis key gone or marked done, fetch from Blob
    blob_client = blob_service.get_blob_client(container=blob_container,
                                               blob=f"{job_id}.json")
    blob_data   = blob_client.download_blob().readall()
    logger.info(f"Fetched final result for job_id={job_id} from Blob")
    return json.loads(blob_data)

def init_population(pop_size: int,
                    num_tasks: int,
                    num_cores: int,
                    rng: random.Random) -> List[List[int]]:
    return [
        [rng.randint(0, num_cores - 1) for _ in range(num_tasks)]
        for _ in range(pop_size)
    ]

def next_generation(population: List[List[int]],
                    fitnesses:   List[float],
                    cfg:         Dict) -> List[List[int]]:
    pop_size = len(population)

    ###############1) Tournament selection
    selected = []
    tour_size = 3
    for _ in range(pop_size):
        aspirants = random.sample(list(zip(population, fitnesses)), tour_size)
        winner    = min(aspirants, key=lambda x: x[1])[0]
        selected.append(winner.copy())

    ###############2) Crossover (uniform one-point)
    offspring = []
    for i in range(0, pop_size, 2):
        p1 = selected[i]
        p2 = selected[(i+1) % pop_size]
        if random.random() < cfg['crossover_rate']:
            pt = random.randint(1, len(p1)-1)
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        else:
            c1, c2 = p1.copy(), p2.copy()
        offspring += [c1, c2]

    ###############3) Mutation
    for ind in offspring:
        if random.random() < cfg['mutation_rate']:
            idx = random.randrange(len(ind))
            ind[idx] = random.randint(0, cfg['num_cores']-1)
    return offspring[:pop_size]

def compute_core_times(individual: List[int],
                       exec_times: List[float],
                       num_cores: int) -> List[float]:
    cores = [0.0]*num_cores
    for i, c in enumerate(individual):
        cores[c] += exec_times[i]
    return cores

async def run_ga(job_id: str, cfg: Dict):
    key     = f"job:{job_id}"

    # Generate task times once
    base_seed = cfg.get("seed") or 0
    rng_exec = random.Random(base_seed)
    exec_times = [rng_exec.randint(1,10) for _ in range(cfg["num_tasks"])]
    logger.info(f"Job {job_id}: generated exec_times (first 5) {exec_times[:5]}")

    # Initialize population
    island_id = int(os.getenv("POD_NAME", "ga-island-0").rsplit("-", 1)[-1])
    pop_seed  = base_seed + island_id
    rng_pop = random.Random(pop_seed)
    population = init_population(cfg["population"], cfg["num_tasks"], cfg["num_cores"], rng=rng_pop)
    logger.debug(f"Job {job_id} Island {island_id}: starting population preview (first 10 individuals): {population[:10]}")

    prev_best      = float('inf')
    best_individual = None

    for gen in range(1, cfg["generations"]+1):
        start = time.time()

        # Parallel fitness eval via evaluator service
        async with httpx.AsyncClient() as client:
            calls = [
                client.post(f"{EVALUATOR_URL}", json={
                    "individual":      indiv,
                    "execution_times": exec_times,
                    "base_energy":     cfg["base_energy"],
                    "idle_energy":     cfg["idle_energy"],
                })
                for indiv in population
            ]
            responses = await httpx.gather(*calls)
        fitnesses = [r.json()["fitness"] for r in responses]

        best = min(fitnesses)
        mean_val = sum(fitnesses)/len(fitnesses)
        # Metrics
        gen_counter.inc()
        best_fitness.set(best)
        mean_fitness.set(mean_val)
        gen_duration.set(time.time() - start)

        logger.info(f"Job {job_id} Gen {gen}: best={best:.4f}, mean={mean_val:.4f}")

        # Overwrite Redis only on improvement
        if best < prev_best:
            idx = fitnesses.index(best)
            best_individual = population[idx]
            rdb.hset(key, mapping={
              "generation": str(gen),
              "best":       str(best)
            })
            prev_best = best

        # Prepare next generation
        population = next_generation(population, fitnesses, cfg)

    # Build final result payload
    final = {
      "best":       prev_best,
      "core_times": compute_core_times(best_individual,
                                       exec_times,
                                       cfg["num_cores"]),
      "metrics": {
        "generations":       cfg["generations"],
        "total_duration_s":  sum(sample.value for sample in gen_duration.collect())
      }
    }

    logger.info(f"Job {job_id} completed: uploading final result")

    # Upload to Azure Blob and clear Redis
    blob_client = blob_service.get_blob_client(blob_container, f"{job_id}.json")
    blob_client.upload_blob(json.dumps(final), overwrite=True)

    rdb.delete(key)
    logger.info(f"Job {job_id}: Redis key cleared")
