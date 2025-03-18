# пример использования MetricEvaluator
# на датсете: data/snils_dataset_annotations.csv
# ответах модели: data/snils_MODELFRAMEWORK_Qwen2-VL-2B-Instruct_VQA_answers_20250124_125639.csv

from metric_evaluator.metric_evaluator import MetricEvaluator

if __name__ == "__main__":
    dataset_annot = "data/snils_dataset_annotations.csv"
    model_answers = (
        "data/snils_MODELFRAMEWORK_Qwen2-VL-2B-Instruct_VQA_answers_20250124_125639.csv"
    )

    metric_eval = MetricEvaluator(dataset_annot, model_answers)

    df_by_id_path_csv = "workspace/ModelsMetrics/df_by_id.csv"
    df_by_id = metric_eval.calculate_metrics_by_id()
    df_by_id.to_csv(
        df_by_id_path_csv,
        sep=";",
        encoding="utf-8-sig",
        index=False,
    )

    df_by_doc_type_path_csv = "workspace/ModelsMetrics/df_by_doc_type.csv"
    df_by_doc_type = metric_eval.calculate_metrics_by_doc_type()
    df_by_doc_type.to_csv(
        df_by_doc_type_path_csv,
        sep=";",
        encoding="utf-8-sig",
        index=False,
    )
    
    df_by_doc_question_path_csv = "workspace/ModelsMetrics/df_by_doc_question.csv"
    df_by_doc_question = metric_eval.calculate_metrics_by_doc_type()
    df_by_doc_question.to_csv(
        df_by_doc_question_path_csv,
        sep=";",
        encoding="utf-8-sig",
        index=False,
    )

    df_general_csv_path = "workspace/ModelsMetrics/df_general.csv"
    df_general = metric_eval.calculate_metrics_general()
    df_general.to_csv(
        df_general_csv_path,
        sep=";",
        encoding="utf-8-sig",
        index=False,
    )
