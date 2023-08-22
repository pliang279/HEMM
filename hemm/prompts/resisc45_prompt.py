class Resisc45Prompt:
    def __init__(self):
        self.prompt = """Image is given to you. Classify if the image belongs to one of the following classes: 
         'basketball_court', 'overpass', 'ground_track_field', 'church', 'chaparral', 'forest', 'parking_lot', 'golf_course', 'baseball_diamond', 
         'meadow', 'beach', 'sparse_residential', 'desert', 'terrace', 'palace', 'bridge', 'commercial_area', 'stadium', 'runway', 'lake', 'railway', 
         'tennis_court', 'ship', 'intersection', 'river', 'freeway', 'airplane', 'industrial_area', 'mountain', 'storage_tank', 'cloud', 'roundabout', 
         'wetland', 'mobile_home_park', 'island', 'harbor', 'railway_station', 'medium_residential', 'sea_ice', 'thermal_power_station', 'snowberg', 
         'circular_farmland', 'airport', 'dense_residential', 'rectangular_farmland'.
          Choose a class from the above classes. 
        """
         
    def format_prompt(self):
        return self.prompt