import pandas as pd
from jiwer import wer, cer
import sacrebleu
import ast


class MetricEvaluator:
    """Класс для вычисления метрик качества предсказаний модели.

    Атрибуты:
        true_csv (pd.DataFrame): DataFrame с правильными ответами.
        pred_csv (pd.DataFrame): DataFrame с предсказаниями модели.
    """

    def __init__(self, true_file: str, prediction_file: str) -> None:
        """Инициализирует экземпляр MetricEvaluator.

        Аргументы:
            true_file (str): Путь к файлу с правильными ответами (CSV или TSV).
            prediction_file (str): Путь к файлу с предсказаниями модели (CSV или TSV).

        Исключения:
            ValueError: Если данные в файлах не совместимы.
        """
        self.true_csv = self.read_file(true_file)
        self.pred_csv = self.read_file(prediction_file)
        self.validate_data()

    def read_file(self, file_path: str) -> pd.DataFrame:
        """Читает файл с определением разделителя (CSV или TSV).

        Аргументы:
            file_path (str): Путь к файлу.

        Возвращает:
            pd.DataFrame: DataFrame с данными из файла.

        Исключения:
            pd.errors.ParserError: Если файл не может быть прочитан как CSV или TSV.
        """
        try:
            return pd.read_csv(file_path, sep=",")  # Читаем как CSV
        except pd.errors.ParserError:
            return pd.read_csv(file_path, sep="\t")  # Если ошибка, читаем как TSV

    def validate_data(self) -> None:
        """Проверяет совместимость данных в true_csv и pred_csv.

        Исключения:
            ValueError: Если столбцы или количество строк не совпадают.
        """
        if self.true_csv.columns.tolist() != self.pred_csv.columns.tolist():
            raise ValueError("Столбцы true_csv и pred_csv не совпадают.")

        if len(self.true_csv) != len(self.pred_csv):
            raise ValueError("Количество строк true_csv и pred_csv не совпадает.")

    def calculate_metrics_by_id(self) -> pd.DataFrame:
        """Вычисляет метрики для каждого ID.

        Возвращает:
            pd.DataFrame: DataFrame с метриками для каждого ID.

        Метрики:
            - WER (Word Error Rate)
            - CER (Character Error Rate)
            - BLEU (Bilingual Evaluation Understudy)
        """
        # Создаем датафрейм для результатов
        result_df = self.true_csv.drop(columns='answear_bbox')  # Удаляем избыточную информацию

        # Создаем хранилища для метрик
        wer_error_list = []
        cer_error_list = []
        bleu_score_list = []

        for row in range(len(self.true_csv)):
            # Преобразуем строку в список
            Y_true = ast.literal_eval(self.true_csv['answers'][row])
            y_pred = ast.literal_eval(self.pred_csv['answers'][row])

            # Проверка на количество ответов
            if len(Y_true) != len(y_pred):
                pass  # Пока игнорируем случаи с разным количеством ответов

            # Вычисляем метрику WER
            wer_error = wer(Y_true, y_pred)
            wer_error_list.append(wer_error)

            # Вычисляем метрику CER
            cer_error = cer(Y_true, y_pred)
            cer_error_list.append(cer_error)

            # Вычисляем метрику BLEU
            bleu_score = sacrebleu.corpus_bleu(Y_true, [y_pred]).score
            bleu_score_list.append(bleu_score)

        # Дополняем результирующий датафрейм метриками
        metrics = {
            'pred_answers': self.pred_csv['answers'],
            'wer_error': wer_error_list,
            'cer_error': cer_error_list,
            'bleu_score': bleu_score_list
        }
        metrics = pd.DataFrame(metrics)
        result_df = pd.concat([result_df, metrics], axis=1)

        return result_df

    def calculate_metrics_by_doc_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Вычисляет метрики для каждого типа документа.

        Аргументы:
            df (pd.DataFrame): DataFrame из метода calculate_metrics_by_id.

        Возвращает:
            pd.DataFrame: DataFrame с метриками для каждого типа документа.
        """
        # Создаем список из всех уникальных типов документов
        doc_types = list(df['doc_class'].value_counts().index)

        # Создаем хранилища для метрик
        wer_error_list = []
        cer_error_list = []
        bleu_score_list = []

        # Фильтруем и аппендим хранилища
        for doc_type in doc_types:
            wer_error_list.append(df[df['doc_class'] == doc_type]['wer_error'].mean())
            cer_error_list.append(df[df['doc_class'] == doc_type]['cer_error'].mean())
            bleu_score_list.append(df[df['doc_class'] == doc_type]['bleu_score'].mean())

        # Создаем DataFrame с метриками для каждого типа документа
        doc_type_metrics = {
            'doc_class': doc_types,
            'wer_error': wer_error_list,
            'cer_error': cer_error_list,
            'bleu_score': bleu_score_list
        }

        return pd.DataFrame(doc_type_metrics)

    def group_by_doc_question(self, df: pd.DataFrame) -> pd.DataFrame:
        """Группирует данные по типу документа и типу вопроса.

        Аргументы:
            df (pd.DataFrame): Исходный DataFrame.

        Возвращает:
            pd.DataFrame: Сгруппированный DataFrame с метриками.
        """
        grouped = df.groupby(['doc_class', 'question_type'])['wer_error', 'cer_error', 'bleu_score'].mean().reset_index()
        return grouped

    def calculate_metrics_general(self) -> dict:
        """Вычисляет общие метрики по всему корпусу данных.

        Возвращает:
            dict: Словарь с метриками WER, CER и BLEU.
        """
        true_answers = []
        pred_answers = []

        for row in range(len(self.true_csv)):
            true_answers.extend(ast.literal_eval(self.true_csv['answers'][row]))
            pred_answers.extend(ast.literal_eval(self.pred_csv['answers'][row]))

        wer_error = wer(true_answers, pred_answers)
        cer_error = cer(true_answers, pred_answers)
        bleu_score = sacrebleu.corpus_bleu(true_answers, [pred_answers]).score

        return {
            'wer_error': wer_error,
            'cer_error': cer_error,
            'bleu_score': bleu_score
        }