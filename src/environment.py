from AirSimClient import *
# from PIL import Image
import time


# Environment interface
class Environment(object):
    def __init__(self):
        super(self).__init__()

    def observe(self):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass


class DroneEnv(Environment):

    def __init__(self):
        super(DroneEnv, self).__init__()
        self.client = MultirotorClient()
        if self.client.confirmConnection():
            print('[INFO] Successfully connected to AirSim.')
        else:
            print('[ERROR] Failed to connect to AirSim')
            return
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.action = []  # todo create a list of action

    # get image from AirSim
    def observe(self, camera_id=0):
        observations = self.client.simGetImages(ImageRequest(camera_id, AirSimImageType.Scene, pixels_as_float=True, compress=True))
        img1d = np.array(observations[0].image_data_float, dtype=np.float)
        img1d = 255/np.max(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, observations[0].height, observations[0].width)
        return img2d

    def reset(self):
        self.client.reset()
        self.client.takeoff()
        return self.observe(camera_id=0)

    def step(self, action):
        self.client.moveByVelocity(self.actions[action])  # todo Change the velocity to current velocity+offset
        time.sleep(0.2)

        position = self.client.getPosition()
        reward, done = self.get_reward(position, self.client.getCollisionInfo().has_collided)
        observation = self.observe(camera_id=0)

        return observation, reward, done

    # Kiran's reward function
    def get_reward(self, position, has_collided):
        pass
