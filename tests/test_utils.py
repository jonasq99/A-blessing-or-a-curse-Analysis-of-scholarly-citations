from unittest.mock import patch
import pandas as pd
import pytest
from src.utils import (
    get_context,
    get_fewshot_cot_examples,
    filter_label,
    calculate_accuracy_per_label,
    llm_label_parser,
    file_finder,
)


# decorator so that pytest runs only this test, used for debugging repo structure, old tests moved to broken_tests folder for debugging
def test_get_context():
    # Test case 1: No [MASK], succeeding context
    text = "Wilhelm II's condemnation of Cecil Rhodes as a 'monstrous villain' in response to the Jameson Raid into Transvaal [CITATION-2] reveals his strong disapproval of Rhodes' actions. This suggests that Wilhelm II held a negative opinion towards Rhodes and his involvement in the scandal."
    expected_context = "Wilhelm II's condemnation of Cecil Rhodes as a 'monstrous villain' in response to the Jameson Raid into Transvaal [CITATION-2]"
    assert get_context(text) == expected_context
    # Test case 2: No [MASK], no succeeding context
    text = 'The end of the nineteenth century marked a significant shift in politics, where political figures had to adapt to the changing landscape of media attention. This shift not only paved the way for the rise of figures like Cecil Rhodes but also laid the foundation for the subsequent criticism of his cult and its association with racism in recent times. As the Times reported, "Among the most picturesque incidents of an age of intercommunication must be reckoned the visit of Mr. Rhodes to Berlin" [CITATION-1].'
    expected_context = 'The end of the nineteenth century marked a significant shift in politics, where political figures had to adapt to the changing landscape of media attention. This shift not only paved the way for the rise of figures like Cecil Rhodes but also laid the foundation for the subsequent criticism of his cult and its association with racism in recent times. As the Times reported, "Among the most picturesque incidents of an age of intercommunication must be reckoned the visit of Mr. Rhodes to Berlin" [CITATION-1].'
    assert get_context(text) == expected_context
    # Test case 3: With [MASK], succeeding context
    text = "be a new land’.[MASK] Thus, as with the stories about his early career and resurrection after the raid, there was the theme of Rhodes having overcome incredible difficulties. For the press, this theme of ‘civilising Africa’ fitted within the broader colonial narrative, which was particularly popular among readers. According to Chamberlain, the Cape-to-Cairo plan even carried ‘sentimental’ meaning for the British public, just as Samoa did for the German one. It is clear that Rhodes and the journalists were both complicit in perpetuating the colonial agenda, using the popular sentiment to further their own interests.[CITATION-95] Rhodes and the journalists benefited from each other: for the journalists, Rhodes provided a way to link the popular colonial theme to power politics, and Rhodes in turn benefited."
    expected_context = "Thus, as with the stories about his early career and resurrection after the raid, there was the theme of Rhodes having overcome incredible difficulties. For the press, this theme of ‘civilising Africa’ fitted within the broader colonial narrative, which was particularly popular among readers. According to Chamberlain, the Cape-to-Cairo plan even carried ‘sentimental’ meaning for the British public, just as Samoa did for the German one. It is clear that Rhodes and the journalists were both complicit in perpetuating the colonial agenda, using the popular sentiment to further their own interests.[CITATION-95]"
    assert get_context(text) == expected_context
    # Test case 4: With [MASK], no succeeding context
    text = "be a new land’.[MASK] Thus, as with the stories about his early career and resurrection after the raid, there was the theme of Rhodes having overcome incredible difficulties. For the press, this theme of ‘civilising Africa’ fitted within the broader colonial narrative, which was particularly popular among readers. According to Chamberlain, the Cape-to-Cairo plan even carried ‘sentimental’ meaning for the British public, just as Samoa did for the German one. It is clear that Rhodes and the journalists were both complicit in perpetuating the colonial agenda, using the popular sentiment to further their own interests.[CITATION-95]"
    expected_context = "Thus, as with the stories about his early career and resurrection after the raid, there was the theme of Rhodes having overcome incredible difficulties. For the press, this theme of ‘civilising Africa’ fitted within the broader colonial narrative, which was particularly popular among readers. According to Chamberlain, the Cape-to-Cairo plan even carried ‘sentimental’ meaning for the British public, just as Samoa did for the German one. It is clear that Rhodes and the journalists were both complicit in perpetuating the colonial agenda, using the popular sentiment to further their own interests.[CITATION-95]"
    assert get_context(text) == expected_context


def test_get_fewshot_cot_examples():
    # Test case 1: Test with provided DataFrame
    df = pd.DataFrame(
        {"input": ["Example 1", "Example 2"], "output": ["Output 1", "Output 2"]}
    )
    expected_output = (
        "Example 1\nReaoning:\nOutput 1\n\nExample 2\nReaoning:\nOutput 2\n\n"
    )
    assert get_fewshot_cot_examples(df) == expected_output

    # Test case 2: Test with empty DataFrame
    df = pd.DataFrame(columns=["input", "output"])
    expected_output = ""
    assert get_fewshot_cot_examples(df) == expected_output


def test_filter_label():
    # Test case 1: Filter label 1 from a single dataframe
    df = pd.DataFrame({"Label": [1, 0, 0]})
    dataframes_dict = {"df1": df}
    label = 1
    expected_result = pd.DataFrame({"Label": [1]})
    assert filter_label(dataframes_dict, label).equals(expected_result)

    # Test case 2: Filter label 1 from multiple dataframes
    df1 = pd.DataFrame({"Label": [1, 0, 1]})
    df2 = pd.DataFrame({"Label": [0, 0, 1]})
    dataframes_dict = {"df1": df1, "df2": df2}
    label = 1
    expected_result = pd.DataFrame({"Label": [1, 1, 1]})
    assert filter_label(dataframes_dict, label).equals(expected_result)

    # Test case 3: Filter label from dataframes with no "Label" column
    df1 = pd.DataFrame({"Value": [1, 0, 1]})
    df2 = pd.DataFrame({"Value": [0, 1, 0]})
    dataframes_dict = {"df1": df1, "df2": df2}
    label = 0
    expected_result = pd.DataFrame()
    assert filter_label(dataframes_dict, label).equals(expected_result)

    # Test case 4: Filter label from empty dataframes
    dataframes_dict = {}
    label = 0
    expected_result = pd.DataFrame()
    assert filter_label(dataframes_dict, label).equals(expected_result)


def test_calculate_accuracy_per_label():
    # testing for None value in predictions
    predictions = [1, 0, 1, 0, 1, None]
    labels = [1, 1, 0, 1, 0, 1]
    label_value = 1
    expected_accuracy = (0.25, 0.33)
    assert (
        round(calculate_accuracy_per_label(predictions, labels, label_value)[0], 2),
        round(calculate_accuracy_per_label(predictions, labels, label_value)[1], 2),
    ) == expected_accuracy

    # testing for None value in predictions and prediction of None label set to opposite of label_value
    predictions = [1, 0, 1, 0, 0, 0]
    labels = [1, 1, 0, 1, None, 1]
    label_value = 1
    expected_accuracy = (0.25, 0.5)
    assert (
        round(calculate_accuracy_per_label(predictions, labels, label_value)[0], 2),
        round(calculate_accuracy_per_label(predictions, labels, label_value)[1], 2),
    ) == expected_accuracy

    # testing for None value in predictions and prediction of None label set to label_value
    predictions = [1, 0, 1, 0, 1, 0]
    labels = [1, 1, 0, 1, None, 1]
    label_value = 1
    expected_accuracy = (0.25, 0.33)
    assert (
        round(calculate_accuracy_per_label(predictions, labels, label_value)[0], 2),
        round(calculate_accuracy_per_label(predictions, labels, label_value)[1], 2),
    ) == expected_accuracy


@patch("src.utils.get_completion_from_messages")
def test_llm_label_parser(mock_get_completion_from_messages):
    # Test case 1: Opinionated citation
    mock_get_completion_from_messages.return_value = "1"
    expected_label = 1
    assert (
        llm_label_parser(mock_get_completion_from_messages.return_value)
        == expected_label
    )

    # Test case 2: Neutral citation
    mock_get_completion_from_messages.return_value = "0"
    expected_label = 0
    assert (
        llm_label_parser(mock_get_completion_from_messages.return_value)
        == expected_label
    )

    # Test case 3: Invalid input
    mock_get_completion_from_messages.return_value = "Invalid input"
    expected_label = None
    assert (
        llm_label_parser(mock_get_completion_from_messages.return_value)
        == expected_label
    )


def test_file_finder():
    # Test case 1: File found
    file_str = "Labels - https___doi.org_10.1093_ehr_cead103.xlsx"
    expected_result = "Au_Nom_de_la_Patrie_Southern_Identities_and_Patriotic_Mobilisation_in_First_World_War_France.json"
    assert file_finder(file_str) == expected_result

    # Test case 2: File not found
    file_str = "Labels - https___doi.org_10.1093_ehr_ceadxxx.xlsx"
    with pytest.raises(FileNotFoundError):
        file_finder(file_str)
