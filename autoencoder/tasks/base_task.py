import os
import luigi

from luigi.task import flatten


class LuigiBaseTask(luigi.Task):
    def run(self) -> None:
        """
        Will need to load correct dataframe that meets all requirements for current task. Then transforms with
        associated transformer. Writes the result dataframe.
        """
        raise NotImplementedError("You need to override this method.")

    def output(self) -> luigi.local_target:
        """
        Will need to output the task result to "intermediate_data" as a joblib file.
        """
        raise NotImplementedError("You need to override this method.")

    def complete(self) -> bool:
        """
        This method changes the default behavior of luigi to assume that if a task is called and is complete,
        all required tasks are complete as well. The desired behavior is that all required tasks are checked and run
        again as well as the tasks depending on them.

        Returns
        -------
        bool
            True if task and requirements are complete, False otherwise
        """
        outputs = flatten(self.output())
        if not all(map(lambda output: output.exists(), outputs)):
            return False
        for task in flatten(self.requires()):
            if not task.complete():
                for output in outputs:
                    if output.exists():
                        output.remove()
                return False
        return True

    def invalidate(self) -> str:
        """
        Deletes the output-file if it exists so when the task gets called it will be run again.

        Returns
        -------
        str
            either message that output was deleted or message that there was nothing to delete
        """
        try:
            os.remove(self.output().path)
            return f"Deleted {self.output().path}"
        except OSError:
            return f"Nothing to delete, {self.output().path} does not exist"
