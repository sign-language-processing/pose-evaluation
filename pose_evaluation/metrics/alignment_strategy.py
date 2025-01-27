
class TrajectoryAligner(PoseProcessor):
    def __init__(self, alignment_strategy:str) -> None:
        self.name = alignment_strategy
    
    def align(self, hyp_trajectory:np.ma.MaskedArray, ref_trajectory:np.ma.MaskedArray) -> Tuple[np.ma.MaskedArray,np.ma.MaskedArray]:
        raise NotImplementedError


    def get_signature(self) -> str:
        return f"alignment_strategy:{self.name}"
    
class DTW_Aligner(TrajectoryAligner):
    def __init__(self) -> None:
        super().__init__(alignment_strategy="dynamic_time_warping")
    
    def align(self, hyp_trajectory:np.ma.MaskedArray, ref_trajectory:np.ma.MaskedArray) -> Tuple[np.ma.MaskedArray,np.ma.MaskedArray]:
        x = hyp_trajectory
        y = ref_trajectory
        _, path = fastdtw(x.data, y.data)  # Use the raw data for DTW computation
    
        # Initialize lists for aligned data and masks
        aligned_x_data = []
        aligned_y_data = []
        
        aligned_x_mask = []
        aligned_y_mask = []
        
        # Loop through the DTW path
        for xi, yi in path:
            # Append aligned data
            aligned_x_data.append(x.data[xi])
            aligned_y_data.append(y.data[yi])
            
            # Append aligned masks (directly use .mask)
            aligned_x_mask.append(x.mask[xi])
            aligned_y_mask.append(y.mask[yi])
        
        # Create aligned masked arrays
        aligned_x = np.ma.array(aligned_x_data, mask=aligned_x_mask)
        aligned_y = np.ma.array(aligned_y_data, mask=aligned_y_mask)
        return aligned_x, aligned_y
        
        
        