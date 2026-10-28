import asyncio
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from typing import Any, List
from src.pipeline import postprocess

@dataclass
class InspectionTask:
    tensor: np.ndarray
    ratio: float
    pad: tuple
    orig_shape: tuple
    future: asyncio.Future

class DynamicMicroBatcher:
    def __init__(self, model_path: str, max_batch_size: int = 4, timeout_ms: float = 15.0):
        self.max_batch_size = max_batch_size
        self.timeout_seconds = timeout_ms / 1000.0
        self.queue: asyncio.Queue[InspectionTask] = asyncio.Queue()
        
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.running = False

    async def start(self):
        self.running = True
        asyncio.create_task(self._batching_loop())

    async def submit(self, tensor: np.ndarray, ratio: float, pad: tuple, orig_shape: tuple) -> Any:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        task = InspectionTask(tensor, ratio, pad, orig_shape, future)
        await self.queue.put(task)
        return await future

    async def _batching_loop(self):
        while self.running:
            tasks: List[InspectionTask] = []
            first_task = await self.queue.get()
            tasks.append(first_task)
            
            deadline = asyncio.get_event_loop().time() + self.timeout_seconds

            while len(tasks) < self.max_batch_size:
                remaining_time = deadline - asyncio.get_event_loop().time()
                if remaining_time <= 0:
                    break
                try:
                    task = await asyncio.wait_for(self.queue.get(), timeout=remaining_time)
                    tasks.append(task)
                except asyncio.TimeoutError:
                    break

            await self._run_inference(tasks)

    async def _run_inference(self, tasks: List[InspectionTask]):
        try:
            batch_tensor = np.stack([t.tensor for t in tasks], axis=0)
            loop = asyncio.get_running_loop()
            raw_outputs = await loop.run_in_executor(
                None, 
                lambda: self.session.run(None, {self.input_name: batch_tensor})
            )
            predictions = raw_outputs[0]

            for i, task in enumerate(tasks):
                res = postprocess(predictions[i], task.ratio, task.pad, task.orig_shape)
                if not task.future.done():
                    task.future.set_result(res)

        except Exception as exc:
            for task in tasks:
                if not task.future.done():
                    task.future.set_exception(exc)
