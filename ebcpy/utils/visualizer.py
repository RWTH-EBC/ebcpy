"""
Module with classes and function to help visualize
different processes inside the framework. Both plots
and print-function/log-function will be implemented here.
The Visualizer Class inherits the Logger class, as logging
will always be used as a default.
"""
import os
from datetime import datetime


class Logger:
    """Base class for showing the process of functions in
    this Framework with print-statements and saving everything
    relevant as a log-file.

    :param str,os.path.normpath cd:
        Directory where to store the output of the Logger and possible
        child-classes. If the given directory can not be created, an error
        will be raised.
    :param str name:
        Name of the reason of logging, e.g. classification, processing etc.
    """

    def __init__(self, cd, name):
        """Instantiate class parameters"""

        self.cd = cd
        if not os.path.isdir(self.cd):
            os.makedirs(self.cd)
        # Setup the logger
        self.filepath_log = os.path.join(cd, "%s.log" % name)
        self.name = name

        # Check if previous logs exist and create some spacer
        _spacer = "-" * 150
        if os.path.isfile(self.filepath_log):
            with open(self.filepath_log, "a+") as log_file:
                log_file.seek(0)
                if log_file.read() != "":
                    log_file.write("\n" + _spacer)

    def log(self, text):
        """
        Logs the given text to the given log.

        :param str text:
            Text to log to the console and file
        """
        print(text)
        datestring = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
        with open(self.filepath_log, "a+") as log_file:
            log_file.write("\n{}: {}".format(datestring, text))

    def get_log_file(self):
        """Function to get the log-file savepath.
        May be used at the end of a process.
        """
        return self.filepath_log
