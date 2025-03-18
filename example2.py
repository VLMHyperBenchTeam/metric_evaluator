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
    df_by_id = metric_eval.save_function_results(
        csv_path=df_by_id_path_csv, func_name="by_id", metrics=["CER"]
    )

    df_by_doc_type_path_csv = "workspace/ModelsMetrics/df_by_doc_type.csv"
    metric_eval.save_function_results(
        csv_path=df_by_doc_type_path_csv,
        func_name="by_doc_type",
        metrics=["CER"],
    )

    df_by_doc_question_path_csv = "workspace/ModelsMetrics/df_by_doc_question.csv"
    metric_eval.save_function_results(
        csv_path=df_by_doc_question_path_csv,
        func_name="by_doc_question",
        metrics=["CER"],
    )

    df_general_csv_path = "workspace/ModelsMetrics/df_general.csv"
    df_general = metric_eval.save_function_results(
        csv_path=df_general_csv_path, func_name="general", metrics=["CER"]
    )
