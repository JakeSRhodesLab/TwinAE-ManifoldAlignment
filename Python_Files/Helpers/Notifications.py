
from pushbullet import Pushbullet
import traceback

def send_pushbullet_notification(api_key, title, message):
    pb = Pushbullet(api_key)
    pb.push_note(title, message)

# Example Usage
api_key = "o.lMfHntQrCefa4z3yY0G8GjxCUE6qJWTQ"

try:
    # Your main script logic here
    print("Running script...")
    # Simulate some work
    send_pushbullet_notification(api_key, "Script Completed", "Your Python script has finished successfully!")
except Exception as e:
    send_pushbullet_notification(api_key, "Script Error", f"Error occurred: {e}")
    traceback.print_exc()  # This will print the traceback to the console
