from model import Model
#import cProfile

def predictor(idx, profile, dt, logging: bool = False, interval: int = 1):
    flight_profile = []
    Model(dt, [profile], flight_profile).altitude_model(logging, interval)
    #profile_name = f"profile_{idx}.prof"
    #cProfile.runctx('Model(dt, [profile], flight_profile).altitude_model(logging, interval)', globals(), locals(), profile_name)
    return flight_profile[0]  # Return the generated profile