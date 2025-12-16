from typing import Dict, List

from apps.visualization.services.aggregator import DataAggregator
from apps.visualization.services.analyzer import DatasetAnalyzer
from apps.visualization.services.chart_recommender import ChartRecommender
from apps.visualization.services.explaination_generator import ExplanationGenerator
from apps.visualization.services.field_selector import FieldSelector
from apps.visualization.services.loader import DatasetLoader

from ..ml.utils import convert_numpy


class VisualizationService:
    def __init__(self):
        self.loader = DatasetLoader()
        self.analyzer = DatasetAnalyzer()
        self.recommender = ChartRecommender(self.analyzer)
        self.field_selector = FieldSelector()
        self.explainer = ExplanationGenerator()
        self.aggregator = DataAggregator()

    def generate_recommendations(self, dataset_model, top_k: int = 3) -> List[Dict]:
        df = self.loader.load_from_model(dataset_model)

        if dataset_model.metadata:
            profile = dataset_model.metadata
        else:
            profile = self.analyzer.analyze(df)
            dataset_model.metadata = convert_numpy(profile)
            print(dataset_model.metadata)
            dataset_model.save(update_fields=["metadata"])

        ranked_charts = self.recommender.recommend(profile, top_k=top_k)

        results = []
        for chart_info in ranked_charts:
            chart_type = chart_info["type"]
            score = chart_info["score"]

            fields = self.field_selector.select_fields(chart_type, profile)

            agg_df, extra_info = self.aggregator.aggregate(df, chart_type, fields)

            explanation = self.explainer.generate(chart_type, fields, score, profile)

            results.append(
                {
                    "chart_type": chart_type,
                    "fields": fields,
                    "aggregated_data": agg_df.to_dict(orient="records"),
                    "reason": explanation["reason"],
                    "detailed_explanation": explanation,
                    "extra_info": extra_info,
                    "suitability_score": score,
                }
            )

        return results
