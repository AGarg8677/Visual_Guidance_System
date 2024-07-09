import abc
import cv2
import numpy as np
class Tracker(abc.ABC):
	@abc.abstractmethod
	def update(self, measurement):
		""" correct tracker, update state """
		pass

	@abc.abstractmethod
	def match(self, measurement):
		""" return a score to order previous RoI matches with detections in next frame """
		pass

	@abc.abstractmethod
	def predict(self):
		""" predict update to measurement in new frame """
		pass

class MOT(abc.ABC):
	@abc.abstractmethod
	def run(self):
		""" process detections in new frame, match with tracked rois, create and delete new trackers """
		pass

class KFTracker(Tracker):
	def __init__(self, label = None, measurement = None, tolerance = 8, score = 0.0, tracker = None, init_frames = None, frame_width = None, frame_height = None, name = None):
		self.name = name # unique ID for this detection 
		self.label = label # store predicted class to improve matching
		self.tolerance = tolerance # number of frames after which unmatched detection is forgotten
		self.score = score # track confidence of detection model
		self.frame_width = frame_width
		self.frame_height = frame_height
		if init_frames:
			self.init_frames = init_frames
		else:
			self.init_frames = int(self.tolerance/2)

		self.measurement = measurement
		self.prediction = None
		self.state = 0 # n when not detected last n frames, 0 when matched last frame, None when tracker is deleted

		if tracker:
			self.tracker = tracker
		else:
			# Kalman filter with state [x, y, vx, vy] and measurement [x, y]
			self.tracker = cv2.KalmanFilter(4,2)

			# measurement is [x,y]
			self.tracker.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32) 
			# transitions: x2 = x1 + dt*vx, y2 = y1 + dt*vy, vx2 = vx1, vy2 = vy1  
			self.tracker.transitionMatrix = np.array([[1,0,0.05,0],[0,1,0,0.05],[0,0,1,0],[0,0,0,1]],np.float32)
			# initially no correlation between noise of dims [x, y, vx, vy]
			self.tracker.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
		if measurement is not None:
			for _ in range(2):
				self.tracker.correct(measurement)
				tracked = self.tracker.predict()
				self.prediction = np.array((tracked[0][0], tracked[1][0]))
				self.predicted_v= np.array((tracked[0][1], tracked[1][1]))

	def update(self, measurement=None, score = None, predict = False, dt=0):

		# if dt is available then update transition matrix otherwise assume zero velocity
		self.tracker.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],np.float32)
		
		if measurement is not None:
			self.state = 0
			self.measurement = measurement
			# print(measurement.shape)
			self.tracker.correct(measurement)
			self.score = score
			if predict:
				tracked = self.tracker.predict()
				self.prediction = np.array((tracked[0][0], tracked[1][0]))
				self.predicted_v= np.array((tracked[0][1], tracked[1][1]))
				return self.prediction
		elif self.state is not None:
			if self.state > self.tolerance-2: # delete object if persistently not detected
				self.state=None
				del self.tracker
				self.tracker = None
				return None
			else:
				self.state +=1
				tracked = self.tracker.predict()
				self.prediction = np.array((tracked[0][0], tracked[1][0]))
				self.predicted_v= np.array((tracked[0][1], tracked[1][1]))
				return self.prediction

	def predict(self, measurement = None, dt = 0):
		self.tracker.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],np.float32)
		if measurement is not None:
			self.tracker.correct(measurement)
		tracked = self.tracker.predict()
		self.prediction = np.array((tracked[0][0], tracked[1][0]))
		self.predicted_v= np.array((tracked[0][1], tracked[1][1]))
		return self.prediction

	def match(self, measurement=None, threshold = None):
		# return negative euclidean distance from center of measurement
		if self.prediction is not None:
			if measurement is not None:
				edf =  -np.linalg.norm(measurement - self.prediction, axis = 1 if len(measurement.shape)>1 else None)
			elif self.measurement is not None:
				edf = -np.linalg.norm(measurement - self.measurement, axis = 1 if len(measurement.shape)>1 else None)
			if threshold:
				return np.where(edf<threshold, -np.inf, edf)
			else:
				return edf
		else:
			return None
	def bounding_box(self, radius_x=30, radius_y=30, frame_width = 1920, frame_height=1080):
		if self.frame_width and self.frame_height:
			frame_width = self.frame_width
			frame_height= self.frame_height
		if self.prediction is not None:
			x_i, y_i = self.prediction
			return [max(x_i-radius_x,0), max(y_i-radius_y,0), min(x_i+radius_x,frame_width), min(y_i+radius_y,frame_height)]
		else:
			return None
	def __repr__(self):
		return f'KFTracker({[self.label, self.measurement, self.tolerance]})'
	def __gt__(self, d):
		return self.score > d
	def __ge__(self, d):
		return self.score >= d
	def __lt__(self, d):
		return self.score < d
	def __le__(self, d):
		return self.score <= d
	def __eq__(self, d):
		return self.score == d
	def __ne__(self, d):
		return self.score != d

class KFSystem(MOT):
	def __init__(self, nms_threshold = None, frame_width = None, frame_height = None, score_threshold = 0.7, tolerance = 4, show_suppressed = False):
		
		self.frame_width = frame_width
		self.frame_height = frame_height
		self.trackers = [] # list of Tracker()s currently active
		self.nms_threshold = nms_threshold # IoU threshold for NMS()
		self.score_threshold = score_threshold # set different threshold for draw()
		self.tolerance = tolerance # number of frames after which unmatched detection is forgotten
		self.show_suppressed = False # show
		self.name = 0 # id given to next new tracked detection, increments by one for each
	def NMS(self, bboxes, labels, scores):
		"""
		Non-maximum supression. Filters multiple detections with similar RoI.
		input:
		bboxes = list(list(float)) of shape [n_detections*2]
		labels = list(str)
		scores = list(float)
		returns bboxes, labels, scores
		"""
		if len(bboxes) == 0:
			return [], [], []
		boxes = np.array(bboxes).astype("float")
		pick = []
		x1 = boxes[:,0]
		y1 = boxes[:,1]
		x2 = boxes[:,2]
		y2 = boxes[:,3]

		area = (x2 - x1 + 1) * (y2 - y1 + 1)
		idxs = np.argsort(scores)

		while len(idxs) > 0:
			last = len(idxs) - 1
			i = idxs[last]
			pick.append(i)
			xx1 = np.maximum(x1[i], x1[idxs[:last]])
			yy1 = np.maximum(y1[i], y1[idxs[:last]])
			xx2 = np.minimum(x2[i], x2[idxs[:last]])
			yy2 = np.minimum(y2[i], y2[idxs[:last]])
			w = np.maximum(0, xx2 - xx1 + 1)
			h = np.maximum(0, yy2 - yy1 + 1)
			overlap = (w * h) / area[idxs[:last]]
			idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > self.nms_threshold)[0])))
		not_picked=[i for i in range(len(scores)) if i not in pick]
		data = [[b, l, s] for b,l, s in zip(boxes[pick].astype("int"), labels [pick], scores[pick])]
		data = sorted(data, key = lambda x: x[2], reverse = True)
		return [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]
	def __repr__(self):
		r = 'KFSystem(['
		offset = ''
		for t in self.trackers:
			r+=offset + t.__repr__()
			offset = ', '
		r += '])'
		return r

	def run(self, bboxes=None, labels=None, scores=None, dt = 0):
		'''
		match detected bboxes in new frame with currently tracked RoIs
		create new RoIs for bbox which doesnt match tracked RoIs
		delete tracked RoIs which have not matched a bbox for self.tolerance frames
		input:
		bboxes = list(list(float)) of shape [n_detections*2]
		labels = list(str)
		scores = list(float)
		returns None 
		'''
		# if there are no detections then predict new position of current tracked objects
		if bboxes is None or not len(bboxes):
			for t in self.trackers:
				t.update(None, t.score, predict = True, dt = dt)
			self.trackers = sorted([t for t in self.trackers if t.tracker is not None], reverse = True)
			return
		if self.nms_threshold:
			bboxes, labels, scores = self.NMS(bboxes, labels, scores)
			bboxes = np.array(bboxes)

		bboxes = np.array([bboxes[:,0] , bboxes[:,1] ], np.float32).transpose()
		nc = len(bboxes)
		detection_matched = [False for _ in self.trackers]
		bbox_matched = [False for _ in bboxes]
		match_scores = []
		for t in self.trackers:
			match_scores.append(t.match(bboxes))


		ms = np.array(match_scores)
		# timer = cv2.getTickCount()
		if self.trackers:
			for i, c in enumerate(bboxes):
				match_index = np.argmax(ms[:, i])
				if not np.isinf(ms[match_index, i]):
					bbox_matched[i] = True

					ms[match_index, :] = [-np.inf for _ in range(nc)]
					self.trackers[match_index].update(c, score = scores[i], predict = True, dt = dt)
					detection_matched[match_index] = True
		# fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
		# print(f'match:{fps}')
		for i, t in enumerate(self.trackers):
			if not detection_matched[i]:
				t.update(None, t.score, dt)
		for i, c in enumerate(bboxes):
			if not bbox_matched[i] and labels[i] =='DRONE':

				self.trackers.append(KFTracker(label = labels[i],
												measurement = c,
												tolerance = self.tolerance,
												score = scores[i],
												frame_width = self.frame_width,
												frame_height = self.frame_height,
												name = self.name))
				self.name+=1
		self.trackers = sorted([t for t in self.trackers if t.tracker is not None], reverse = True)

	def draw(self, frame, draw_fn=None, color = False, return_drawn_frame = False):
		if draw_fn:
			

			base_threshold = self.score_threshold*self.trackers[0].score
			for t in self.trackers:
				if t.score > base_threshold:
					bb = t.bounding_box(frame_width = self.frame_width, frame_height = self.frame_height)
					if bb:
						if color:
							frame = draw_fn(frame, bb, (0,255,0))
						else:
							frame = draw_fn(frame, bb)
				elif self.show_suppressed:
					bb = t.bounding_box(frame_width = self.frame_width, frame_height = self.frame_height)
					if bb:
						if color:
							draw_fn(frame, bb, (0,255,0))
						else:
							draw_fn(frame, bb)
			if return_drawn_frame:
				return frame

		else:
			raise NotImplementedError('Provide a draw_fn with arguments frame, bounding_box [and color]')
