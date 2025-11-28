#! /usr/bin/python3
import time
import os, sys
from datetime import datetime, timedelta

LIB_NAME="libfastrrt.so"

def exec(cmd: str) -> bool:
    result = os.system(cmd)
    return result == 0

def beep_error():
    playsound('sounds/alert.wav')

def beep_ok():
    # For Linux systems, we can use the "beep" command or the ASCII Bell (Ctrl+G)
    playsound('sounds/build_ok.wav')


def playsound(sound, block = True):
    """Play a sound using GStreamer.

    Inspired by this:
    https://gstreamer.freedesktop.org/documentation/tutorials/playback/playbin-usage.html
    """
    # pathname2url escapes non-URL-safe characters
    from os.path import abspath, exists
    try:
        from urllib.request import pathname2url
    except ImportError:
        # python 2
        from urllib import pathname2url

    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst

    Gst.init(None)

    playbin = Gst.ElementFactory.make('playbin', 'playbin')
    if sound.startswith(('http://', 'https://')):
        playbin.props.uri = sound
    else:
        path = abspath(sound)
        if not exists(path):
            return
        playbin.props.uri = 'file://' + pathname2url(path)


    set_result = playbin.set_state(Gst.State.PLAYING)
    if set_result != Gst.StateChangeReturn.ASYNC:
        return

    if block:
        bus = playbin.get_bus()
        try:
            bus.poll(Gst.MessageType.EOS, Gst.CLOCK_TIME_NONE)
        finally:
            playbin.set_state(Gst.State.NULL)
    


def print_check_time(time_delta_sec: int):
    # Get the current time
    current_time = datetime.now()
        # Add one minute
    new_time = current_time + timedelta(seconds=time_delta_sec)
    # Print the new time
    print("Next check:", new_time.strftime("%H:%M:%S"))

def prepare_system_lib() -> bool:
    res = exec("cd .. && make local")    
    if not res:
        print(f"error compiling {LIB_NAME} even though the cpp tests are ok")
    return res

def execute_tests() -> bool:
    cpp_tests = exec("make test")
    if not cpp_tests:
        return False
    if not prepare_system_lib():
        return False

    return exec("cd python && sh ./run_py_tests.sh")


def main():
    broken_tests = False
    print ("Continous testing tool for Makefile projects")
    print ("Makefile must have: make test")
    print ("final test file must be named unittest")


    broken_tests = False
   
    while True:
        if not execute_tests():
            beep_error()
            broken_tests = True
        else:
            if broken_tests:
                beep_ok()
            
            exec("clear")
            print("libfastrrt is ok")
            broken_tests = False
        
        print_check_time(60)
        time.sleep(60)

if __name__ == "__main__":
    main()
