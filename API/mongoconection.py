from pymongo import MongoClient

client= MongoClient("mongodb+srv://<Hyper Gamex>:<0jaQsgPFiRj5K5QKd>@cluster0.mongodb.net/test?retryWrites=true&w=majority")
db=client["escuela_db"]
collection=db["estudiantes"]
