from tracker import *

def select_box(event, x, y, flags, param):
    # global selected_box, box_selected, tracker, frame, tracking_enabled, detection_mode
    


    if event == cv2.EVENT_LBUTTONDOWN:
    	match_dist = -np.linalg.norm(np.array([x,y]) - np.array([t.prediction if t.prediction is not None else np.inf for t in param['system'].trackers ]), axis = 1 if len(measurement.shape)>1 else None)
    	param['system'].clicked_tracker = param['system'].trackers[np.argmax(match_dist)]


class MOTGUI(KFSystem):

	def __init__(self, nms_threshold = None, frame_width = None, frame_height = None, score_threshold = 0.7, tolerance = 4, show_suppressed = False):
		super.__init__(nms_threshold, 
			frame_width, 
			frame_height, 
			score_threshold, 
			tolerance, 
			show_suppressed)
		self.clicked_tracker = None
		cv2.namedWindow("Object Tracking")
		cv2.setMouseCallback("Object Tracking", select_box, {'system':self})
	


	def draw(self,frame, draw_fn=None, color=False, return_drawn_frame=False):

		frame = super.draw(frame,
			draw_fn,
			color,
			True)
		if self.clicked_tracker is not None:
			if self.clicked_tracker.tracker is not None:
				x = self.clicked_tracker.prediction.astype(int)
				frame = cv2.arrowedLine(frame,x, (x[0]+self.clicked_tracker.predicted_v[0], x[1]+self.clicked_tracker.predicted_v[1]), (255,0,0), 8)
			else:
				self.clicked_tracker = None

