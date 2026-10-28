import asyncio
import time
import httpx

TARGET_URL = "http://127.0.0.1:8000/v1/inspect"
TEST_IMAGE_PATH = "test.jpg"
NUM_REQUESTS = 60
CONCURRENCY = 6

async def send_request(client: httpx.AsyncClient, image_bytes: bytes):
    files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
    start = time.perf_counter()
    resp = await client.post(TARGET_URL, files=files)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return resp.status_code, latency_ms

async def main():
    with open(TEST_IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    async with httpx.AsyncClient(limits=limits, timeout=60.0) as client:
        # Warmup pass
        await send_request(client, image_bytes)

        print(f"Firing {NUM_REQUESTS} concurrent requests across {CONCURRENCY} workers...")
        start_wall = time.perf_counter()
        tasks = [send_request(client, image_bytes) for _ in range(NUM_REQUESTS)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_wall

    latencies = [lat for code, lat in results if code == 200]
    latencies.sort()

    print("\n================ SYSTEM BENCHMARK SUMMARY ================")
    print(f"Total Requests:      {NUM_REQUESTS}")
    print(f"Successful (200 OK): {len(latencies)}")
    print(f"Total Wall Clock:    {total_time:.2f}s")
    print(f"Throughput:          {len(latencies) / total_time:.2f} req/s (FPS)")
    print(f"P50 Latency:         {latencies[len(latencies)//2]:.2f} ms")
    print(f"P95 Latency:         {latencies[int(len(latencies)*0.95)]:.2f} ms")
    print(f"P99 Latency:         {latencies[int(len(latencies)*0.99)]:.2f} ms")
    print("==========================================================")

if __name__ == "__main__":
    asyncio.run(main())
