import torch
import torchvision.models as models
import time
import torch._dynamo as dynamo

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def reset_environment():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.manual_seed(42)
    dynamo.reset()

def measure_densenet_performance():
    
    model = models.densenet121(weights="DEFAULT").to(device).eval()
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    
    print("=" * 60)
    print("DenseNet121 Performance Measurement")
    print("=" * 60)
    
    print("\n1. Eager Mode Baseline:")
    reset_environment()
    
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_tensor)
        
        start_time = time.time()
        for _ in range(100):
            _ = model(input_tensor)
        eager_time = time.time() - start_time
        
    print(f"   Eager inference time (100 runs): {eager_time:.4f}s")
    print(f"   Average per inference: {eager_time/100:.6f}s")
    
    print("\n2. Compile Time Measurement:")
    reset_environment()
    
    with torch.no_grad():
        start_compile = time.time()
        compiled_model = torch.compile(model, mode="default")
        
        _ = compiled_model(input_tensor)
        compile_time = time.time() - start_compile
        
    print(f"   Total compile time (including first inference): {compile_time:.4f}s")
    
    print("\n3. Compiled Inference Time:")
    
    with torch.no_grad():
        for _ in range(5):
            _ = compiled_model(input_tensor)
        
        start_time = time.time()
        for _ in range(100):
            _ = compiled_model(input_tensor)
        compiled_inference_time = time.time() - start_time
        
    print(f"   Compiled inference time (100 runs): {compiled_inference_time:.4f}s")
    print(f"   Average per inference: {compiled_inference_time/100:.6f}s")
    
    print("\n4. Guard Time Analysis:")
    print("   Testing with different input shapes to trigger guards...")
    
    test_shapes = [(1, 3, 224, 224), (2, 3, 224, 224), (1, 3, 256, 256)]
    
    for i, shape in enumerate(test_shapes):
        try:
            test_input = torch.randn(*shape).to(device)
            
            with torch.no_grad():
                start_time = time.time()
                for _ in range(10):
                    _ = compiled_model(test_input)
                shape_time = time.time() - start_time
                
            print(f"   Shape {shape}: {shape_time:.4f}s (10 runs), avg: {shape_time/10:.6f}s")
            
        except Exception as e:
            print(f"   Shape {shape}: Failed - {e}")
    
    print("\n5. Backend Comparison:")
    
    backends = ["inductor", "aot_eager"]
    
    for backend in backends:
        try:
            reset_environment()
            print(f"\n   Testing with backend: {backend}")
            
            with torch.no_grad():
                start_time = time.time()
                backend_compiled_model = torch.compile(model, backend=backend)
                _ = backend_compiled_model(input_tensor)
                backend_compile_time = time.time() - start_time
                
                start_time = time.time()
                for _ in range(50):
                    _ = backend_compiled_model(input_tensor)
                backend_inference_time = time.time() - start_time
                
            print(f"     Compile time: {backend_compile_time:.4f}s")
            print(f"     Inference time (50 runs): {backend_inference_time:.4f}s")
            print(f"     Average per inference: {backend_inference_time/50:.6f}s")
            
        except Exception as e:
            print(f"     Backend {backend} failed: {e}")
    
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Eager mode avg per inference:     {eager_time/100:.6f}s")
    print(f"Compiled mode avg per inference:  {compiled_inference_time/100:.6f}s")
    print(f"Compile time overhead:            {compile_time:.4f}s")
    
    if compiled_inference_time > 0:
        speedup = eager_time / compiled_inference_time
        print(f"Speedup factor:                   {speedup:.2f}x")
        
        time_saved_per_inference = (eager_time/100) - (compiled_inference_time/100)
        if time_saved_per_inference > 0:
            break_even = compile_time / time_saved_per_inference
            print(f"Break-even point:                 {break_even:.0f} inferences")
        else:
            print("Break-even point:                 Never (compiled is slower)")
    
    print("\n7. Memory Usage:")
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(input_tensor)
            eager_memory = torch.cuda.max_memory_allocated()
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = compiled_model(input_tensor)
            compiled_memory = torch.cuda.max_memory_allocated()
        
        print(f"   Eager mode peak memory:    {eager_memory / 1024**2:.2f} MB")
        print(f"   Compiled mode peak memory: {compiled_memory / 1024**2:.2f} MB")
        print(f"   Memory overhead:           {(compiled_memory - eager_memory) / 1024**2:.2f} MB")
    else:
        print("   Memory measurement only available on CUDA")

if __name__ == "__main__":
    measure_densenet_performance()