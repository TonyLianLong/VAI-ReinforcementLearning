import imageio
import os


class VideoRecorder(object):
    def __init__(self, dir_name, height=100, width=100, camera_id=0, fps=25):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env, losses=[]):
        if self.enabled:
            frame = env.render(
                mode='rgb_array',
                height=self.height,
                width=self.width,
                camera_id=self.camera_id
            )
            if 'video' in env._mode:
                greenscreen = env.env.env.env
                assert hasattr(greenscreen, "green_screen_wrapper")
                frame = greenscreen.apply_to(frame)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
