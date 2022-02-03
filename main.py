import numpy as np
import pandas as pd
import uuid
from sklearn.metrics import classification_report
from IPython.display import display, HTML

def not_equal_series(se_a: pd.Series, se_b: pd.Series) -> pd.Series:
    """
    Get a boolean selector that selects differences in two series.

    We are careful working with missing values, as NaN == NaN resolves to
    False.
    """
    fill = str(uuid.uuid4())
    return se_a.fillna(fill) != se_b.fillna(fill)


class Inspector:
    """

    Given a clean column, a dirty column and a column with cleaning
    suggestions, Inspector creates a cleaning report or lets you inspect
    single cleaning suggestions and their contexts more closely.

    All series are expected to be sorted in the same way. The series' index
    identifies each value uniquely.

    :param y_clean: pandas series containing the cleaned data (ground truth).
    :param y_dirty: pandas series containing the dirty data.
    :param y_pred: pandas series containing the cleaning suggestions.
    "param assume_errors_known: boolean, if true then we assume that all error
                                positions are known upfront.

    """
    def __init__(self,
                 y_clean: pd.Series,
                 y_dirty: pd.Series,
                 y_pred: pd.Series,
                 assume_errors_known: bool = True):
        self.assume_errors_known = assume_errors_known

        # "true" error positions in y_dirty according to y_clean
        self._error_positions = not_equal_series(y_clean, y_dirty)

        # predicted error positions in y_dirty according to y_pred
        self._predicted_error_positions = not_equal_series(y_dirty, y_pred)

        # positions where the data has been incorrectly cleaned
        self._cleaning_error_positions = not_equal_series(y_clean, y_pred)

    def cleaning_report(self):
        """
        A full report on the cleaning task, inspired by sklearn's
        classification_report.
        """
        pass

    def inspect_cleaning_results(self,
                                 df_clean: pd.DataFrame,
                                 df_pred: pd.DataFrame,
                                 df_dirty: pd.DataFrame,
                                 context_col_selector,
                                 context_height: int = 3):
        """
        A magical function that integrates with IPython to make inspecting
        cleaning mistakes more fun.
        """
        error_indices = self._cleaning_error_positions.index[self._cleaning_error_positions == True]
        for i, pos in enumerate(error_indices):
            print(f"Evaluating error {i+1} from {len(error_indices)}")
            print(f"Error in row {pos}:")

            row_start, row_end = pos-context_height, pos+context_height
            highlight_row = lambda x: ['background: lightgreen' if x.name == pos
                else '' for _ in x]
            clean_context = df_clean.iloc[row_start:row_end, :].loc[:, context_col_selector]
            dirty_context = df_dirty.iloc[row_start:row_end, :].loc[:, context_col_selector]
            pred_context = df_pred.iloc[row_start:row_end, :].loc[:, context_col_selector]

            clean_context = clean_context.style.apply(highlight_row, axis=1).to_html()
            dirty_context = dirty_context.style.apply(highlight_row, axis=1).to_html()
            pred_context = pred_context.style.apply(highlight_row, axis=1).to_html()

            display(HTML('<hr>'))
            display(HTML('<h3>Clean Data</h3>'))
            display(HTML(clean_context))
            display(HTML('<h3>Dirty Data</h3>'))
            display(HTML(dirty_context))
            display(HTML('<h3>Predicted Data</h3>'))
            display(HTML(pred_context))
            display(HTML('<hr>'))
            wants_more = input('Enter to continue, any input to abort')
            if wants_more != '':
                break



    def error_cleaning_performance(self,
                             y_clean: pd.Series,
                             y_pred: pd.Series,
                             y_dirty: pd.Series):
        """
        Calculate the f1-score between the clean labels and the predicted
        labels.

        As defined by Rekasinas et al. 2017 (Holoclean), we compute:
        - Precision as the fraction of correct repairs over the total number
          of repairs performed.
        - Recall as the fraction of (correct repairs of real errors) over the
          total number of errors.

        Most data-cleaning publications work under the assumption that all
        errors have been successfully detected. (Mahdavi 2020) This behavior
        can be controlled with the parameter assume_errors_known. If we work
        under this assumptions, true negatives and false positives become
        impossible.
        """
        if self.assume_errors_known:
            y_clean = y_clean.loc[self._error_positions]
            y_pred = y_pred.loc[self._error_positions]
            y_dirty = y_dirty.loc[self._error_positions]

        tp = sum(np.logical_and(y_dirty != y_clean, y_pred == y_clean))
        fp = sum(np.logical_and(y_dirty == y_clean, y_pred != y_clean))
        fn = sum(np.logical_and(y_dirty != y_clean, y_pred != y_clean))
        tn = sum(np.logical_and(y_dirty == y_clean, y_pred == y_clean))

        print("Calculating Cleaning Performance.")
        print(f"Counted {tp} TPs, {fp} FPs, {fn} FNs and {tn} TNs.")

        p = .0 if (tp + fp) == 0 else tp / (tp + fp)
        r = .0 if (tp + fn) == 0 else tp / (tp + fn)
        f1_score = .0 if (p+r) == 0 else 2 * (p*r)/(p+r)
        return f1_score


    def error_detection_performance(self):
        """
        Calculate the f1-score for finding the correct position of errors in
        y_dirty.
        """
        return classification_report(self._error_positions,
                                     self._predicted_error_positions)
