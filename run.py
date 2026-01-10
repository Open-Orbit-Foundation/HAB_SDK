from model import Model
#import cProfile

def predictor(idx, profile, dt, logging: bool = False, interval: int = 1):
    flight_profile = []
    Model(dt, [profile], flight_profile).altitude_model(logging, interval)
    #profile_name = f"profile_{idx}.prof"
    #cProfile.runctx('Model(dt, [profile], flight_profile).altitude_model(logging, interval)', globals(), locals(), profile_name)
    return flight_profile[0] if flight_profile else None # Return the generated profile

def predictor_batch(batch, dt, logging: bool = False, interval: int = 1):
    """
    batch: list of (idx, MissionProfile)
    returns: list of (idx, FlightProfile|None) in the same order as batch
    """
    idxs = [i for i, _ in batch]
    profiles = [p for _, p in batch]

    results = []
    Model(dt, profiles, results).altitude_model(logging, interval)

    # results aligns with profiles order
    return list(zip(idxs, results))