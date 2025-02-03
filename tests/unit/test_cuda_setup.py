import unittest
import torch

class TestCudaSetup(unittest.TestCase):
    def test_cuda_availability(self):
        """Check if CUDA is recognized by PyTorch"""
        self.assertTrue(torch.cuda.is_available(), 
                       "CUDA not available. Check installation and driver compatibility.")
    def test_cuda_device_count(self):
        """
        Test that at least one CUDA-capable device is detected.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available, skipping device count test.")
        device_count = torch.cuda.device_count()
        self.assertGreater(
            device_count, 0,
            f"Expected at least 1 CUDA device, but found {device_count}."
        )

    def test_basic_gpu_operations(self):
        """Verify basic tensor operations on GPU"""
        if not torch.cuda.is_available():
            self.skipTest("Skipping GPU test - CUDA not available")
            
        device = torch.device('cuda:0')
        
        # Test tensor creation and movement
        tensor = torch.tensor([1.0, 2.0]).to(device)
        self.assertEqual(tensor.device, device, 
                        "Tensor not moved to GPU device")

        # Test basic computation
        result = tensor * 2
        expected = torch.tensor([2.0, 4.0], device=device)
        self.assertTrue(torch.allclose(result, expected),
                       "GPU computation produced unexpected results")

    def test_version_info(self):
        """Print PyTorch and CUDA version info for diagnostics."""
        print("\n===== System Information =====")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA build version: {torch.version.cuda}")
        if torch.cuda.is_available():
            try:
                runtime_version = torch.cuda.runtime_version()
                print(f"CUDA runtime version: {runtime_version}")
            except Exception as e:
                print(f"Could not determine CUDA runtime version: {e}")
            # Print properties for each CUDA device
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(
                    f"GPU {i}: {props.name} "
                    f"(Compute Capability {props.major}.{props.minor}, "
                    f"Total Memory: {props.total_memory / (1024**3):.2f} GB)"
                )
        else:
            print("CUDA is not available.")
        print("==============================\n")

if __name__ == '__main__':
    unittest.main(verbosity=2)