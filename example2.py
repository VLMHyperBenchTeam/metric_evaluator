# пример использования MetricEvaluator
# на датсете: data/snils_dataset_annotations.csv
# ответах модели: data/snils_MODELFRAMEWORK_Qwen2-VL-2B-Instruct_VQA_answers_20250124_125639.csv

from metric_evaluator.metric_evaluator import MetricEvaluator

if __name__ == "__main__":
    dataset_annot = "data/snils_dataset_annotations.csv"
    model_answers = (
        "data/snils_MODELFRAMEWORK_Qwen2-VL-2B-Instruct_VQA_answers_20250124_125639.csv"
    )
    metrics_aggregators = [
        "by_id",
        "by_doc_question",
        "by_doc_type",
        "general",
    ]
    metrics=["CER"]

    metric_eval = MetricEvaluator(dataset_annot, model_answers)
    
    # Производим расчет метрик по всем агрегаторам
    for metrics_aggregator in metrics_aggregators:
        metric_csv_path = f"workspace/ModelsMetrics/df_{metrics_aggregator}.csv"
        metric_eval.save_function_results(
        csv_path=metric_csv_path, func_name=metrics_aggregator, metrics=metrics
    )
