"""
Locust load testing script for ML Image Classification API
Simulates flood of requests to test model performance under load
"""

from locust import HttpUser, task, between
import os
import random
import io

class MLAPILoadTest(HttpUser):
    """Load testing user for ML API endpoints"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Called when a simulated user starts"""
        # Health check
        self.client.get("/api/health")
    
    @task(5)
    def predict_single_image(self):
        """Test single image prediction endpoint"""
        # Create a dummy image file for testing
        # In production, you'd use actual image files
        try:
            # Try to find a test image or create a dummy one
            test_image_path = self._get_test_image()
            
            if test_image_path and os.path.exists(test_image_path):
                with open(test_image_path, 'rb') as img:
                    files = {'image': ('test.jpg', img, 'image/jpeg')}
                    with self.client.post(
                        "/api/predict",
                        files=files,
                        catch_response=True,
                        name="Predict Single Image"
                    ) as response:
                        if response.status_code == 200:
                            result = response.json()
                            response.success()
                        elif response.status_code == 503:
                            response.failure("Model not loaded")
                        else:
                            response.failure(f"Unexpected status: {response.status_code}")
            else:
                # Create a minimal dummy image for testing
                dummy_image = self._create_dummy_image()
                files = {'image': ('test.jpg', dummy_image, 'image/jpeg')}
                self.client.post(
                    "/api/predict",
                    files=files,
                    name="Predict Single Image (Dummy)"
                )
        except Exception as e:
            print(f"Error in predict_single_image: {e}")
    
    @task(2)
    def get_model_uptime(self):
        """Test model uptime endpoint"""
        with self.client.get(
            "/api/model/uptime",
            catch_response=True,
            name="Get Model Uptime"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(3)
    def get_health(self):
        """Test health check endpoint"""
        with self.client.get(
            "/api/health",
            name="Health Check"
        ) as response:
            pass
    
    @task(1)
    def get_stats(self):
        """Test stats endpoint"""
        with self.client.get(
            "/api/stats",
            catch_response=True,
            name="Get Stats"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(1)
    def batch_predict(self):
        """Test batch prediction endpoint"""
        try:
            dummy_image = self._create_dummy_image()
            files = [
                ('images', ('test1.jpg', dummy_image, 'image/jpeg')),
                ('images', ('test2.jpg', dummy_image, 'image/jpeg')),
            ]
            with self.client.post(
                "/api/predict/batch",
                files=files,
                catch_response=True,
                name="Batch Predict"
            ) as response:
                if response.status_code in [200, 400]:
                    response.success()
                else:
                    response.failure(f"Status: {response.status_code}")
        except Exception as e:
            print(f"Error in batch_predict: {e}")
    
    def _get_test_image(self):
        """Try to find a test image file"""
        test_paths = [
            'data/test',
            'data/train',
            'static',
        ]
        
        for path in test_paths:
            if os.path.exists(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            return os.path.join(root, file)
        return None
    
    def _create_dummy_image(self):
        """Create a minimal valid JPEG image for testing"""
        # Minimal valid 1x1 JPEG
        dummy_jpeg = bytes([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46,
            0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
            0xFF, 0xDB, 0x00, 0x43, 0x00, 0x08, 0x06, 0x06, 0x07, 0x06,
            0x05, 0x08, 0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C,
            0xFF, 0xD9
        ])
        return io.BytesIO(dummy_jpeg)

