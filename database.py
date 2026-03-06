from pymongo import MongoClient
import datetime

# 1. Connect to your local MongoDB
# 'localhost:27017' is the default for MongoDB Compass
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    db = client["weather_intelligence"]  # Database name
    collection = db["history"]            # Collection name
    
    # Test connection
    client.server_info() 
    print("✅ MongoDB Connection Successful!")
except Exception as e:
    print(f"❌ MongoDB Connection Failed: {e}")

def save_prediction(input_data, cluster, prediction):
    """
    Saves the weather reading and the model's result to Mongo.
    """
    document = {
        "timestamp": datetime.datetime.now(),
        "input_features": input_data, # Dictionary of the 13 features
        "cluster_assigned": int(cluster),
        "visibility_prediction": float(round(prediction, 2)),
        "unit": "statute miles"
    }
    
    result = collection.insert_one(document)
    return str(result.inserted_id)