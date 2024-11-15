class Lane(object):
    def __init__(self, id=-1, width=3, list=[], merge_end_point= [], s_map=[], angle_list = []):
        super(Lane, self).__init__()
        self.id = id
        self.width = width
        self.list = list # vector of Point2D
        self.s_map = s_map
        self.merge_end_point = merge_end_point
        self.angle_list = angle_list 
