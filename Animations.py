class MoveAnimation:
    def __init__(self, start_time=0, duration = -1, start_pos:tuple=(0,0), end_pos:tuple=(0,0), start_size:tuple=(-1,-1), end_size :tuple=(-1,-1)):
        self.start_time = start_time
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.start_size = start_size
        self.end_size = end_size
        self.duration = duration
    
    
    def get_pos(self, t):
        return (self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * t / self.duration,
                self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * t / self.duration)
        
    def get_size(self, t):
        return (1+self.start_size[0] + (self.end_size[0] - self.start_size[0]) * t / self.duration,
                1+self.start_size[1] + (self.end_size[1] - self.start_size[1]) * t / self.duration)
        




