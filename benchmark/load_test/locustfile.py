from locust import HttpUser, task, between
import random
import json

class SolanaValidatorUser(HttpUser):
    wait_time = between(0.1, 0.5)
    
    @task
    def send_transaction(self):
        # Generate mock transaction data
        tx_data = {
            "instructions": [{
                "program_id": str(random.getrandbits(256)),
                "accounts": [str(random.getrandbits(256)) for _ in range(3)],
                "data": bytes([random.randint(0, 255) for _ in range(32)]).hex()
            }],
            "signatures": [str(random.getrandbits(512))],
            "recent_blockhash": str(random.getrandbits(256))
        }
        
        headers = {"Content-Type": "application/json"}
        self.client.post(
            "/transactions", 
            data=json.dumps(tx_data),
            headers=headers
        )
