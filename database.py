from flask_pymongo import pymongo

CONNECTION_STRING = "mongodb+srv://biya:rabeea190@cluster0.ewzs0.mongodb.net/OutfitAdobe?retryWrites=true&w=majority"
client = pymongo.MongoClient(CONNECTION_STRING,tls=True,tlsAllowInvalidCertificates=True)
db = client.get_database('OutfitAdobe')
print(db)
measurements = pymongo.collection.Collection(db, 'measuremnets')
