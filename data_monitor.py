import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

class FileWatchHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('loan.csv'):
            print(f"{event.src_path} has been modified. Triggering the pipeline.")
            subprocess.run(['python', 'pipeline.py'])

def start_watcher():
    path = '../data/raw/'
    event_handler = FileWatchHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    print("Started watcher on data/raw/loan.csv")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == '__main__':
    start_watcher()
