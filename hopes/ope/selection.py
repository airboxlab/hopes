from dataclasses import asdict

import pandas as pd

from hopes.ope.evaluation import OffPolicyEvaluationResults


class OffPolicySelection:
    @staticmethod
    def select_top_k(
        evaluation_results: list[OffPolicyEvaluationResults],
        metric: str = "mean",
        top_k: int = 1,
    ) -> list[OffPolicyEvaluationResults]:
        """Select the top-k policies based on the given metric.

        :param evaluation_results: The results of the off-policy evaluation for multiple
            policies.
        :param metric: The metric to use for the selection. Can be "mean", "lower_bound",
            "upper_bound".
        :param top_k: The number of policies to select.
        :return: The top-k policies based on the given metric.
        """
        first_result = evaluation_results[0]
        estimator_names = first_result.results.keys()
        policy_names = [result.policy_name for result in evaluation_results]
        significance_level = first_result.significance_level

        # gather the results in a dataframe
        df = pd.DataFrame(
            columns=["mean", "std", "lower_bound", "upper_bound"],
            index=pd.MultiIndex.from_tuples(zip(policy_names, estimator_names)),
        )
        for policy_results in evaluation_results:
            for estimator_name, result in policy_results.results.items():
                df.loc[(policy_results.policy_name, estimator_name), :] = asdict(result)

        top_k_policies = (
            # group by policies
            df.groupby(by=df.index.get_level_values(0))
            # mean of each metric of each policy for all estimators
            .mean()
            # sort by the metric
            .sort_values(by=metric, ascending=False)
            # sample top k
            .head(top_k)
        )

        # get the results of the top k policies
        top_k_results = [
            OffPolicyEvaluationResults(
                policy_name=policy_name,
                results=evaluation_results[policy_names.index(policy_name)].results,
                significance_level=significance_level,
            )
            for policy_name in top_k_policies.index
        ]

        return top_k_results
