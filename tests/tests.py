import sys
from pathlib import Path
import pytest

from src.utils import get_context
from src.data_creator import create_data


# decorator so that pytest runs only this test, used for debugging repo structure, old tests moved to broken_tests folder for debugging
@pytest.mark.correct
def test_get_context():
    # Test case 1: No [MASK], succeeding context
    text = "Wilhelm II's condemnation of Cecil Rhodes as a 'monstrous villain' in response to the Jameson Raid into Transvaal [CITATION-2] reveals his strong disapproval of Rhodes' actions. This suggests that Wilhelm II held a negative opinion towards Rhodes and his involvement in the scandal."
    expected_context = "Wilhelm II's condemnation of Cecil Rhodes as a 'monstrous villain' in response to the Jameson Raid into Transvaal [CITATION-2]"
    assert get_context(text) == expected_context
    # Test case 2: No [MASK], no succeeding context
    text = (
        "The end of the nineteenth century marked a significant shift in politics, where political figures had to adapt to the changing landscape of media attention. This shift not only paved the way for the rise of figures like Cecil Rhodes but also laid the foundation for the subsequent criticism of his cult and its association with racism in recent times. As the Times reported, "
        "Among the most picturesque incidents of an age of intercommunication must be reckoned the visit of Mr. Rhodes to Berlin"
        " [CITATION-1]."
    )
    expected_context = (
        "The end of the nineteenth century marked a significant shift in politics, where political figures had to adapt to the changing landscape of media attention. This shift not only paved the way for the rise of figures like Cecil Rhodes but also laid the foundation for the subsequent criticism of his cult and its association with racism in recent times. As the Times reported, "
        "Among the most picturesque incidents of an age of intercommunication must be reckoned the visit of Mr. Rhodes to Berlin"
        " [CITATION-1]."
    )
    assert get_context(text) == expected_context

    # Test case 3: With [MASK], succeeding context
    text = "be a new land’.[MASK] Thus, as with the stories about his early career and resurrection after the raid, there was the theme of Rhodes having overcome incredible difficulties. For the press, this theme of ‘civilising Africa’ fitted within the broader colonial narrative, which was particularly popular among readers. According to Chamberlain, the Cape-to-Cairo plan even carried ‘sentimental’ meaning for the British public, just as Samoa did for the German one. It is clear that Rhodes and the journalists were both complicit in perpetuating the colonial agenda, using the popular sentiment to further their own interests.[CITATION-95] Rhodes and the journalists benefited from each other: for the journalists, Rhodes provided a way to link the popular colonial theme to power politics, and Rhodes in turn benefited."
    expected_context = "Thus, as with the stories about his early career and resurrection after the raid, there was the theme of Rhodes having overcome incredible difficulties. For the press, this theme of ‘civilising Africa’ fitted within the broader colonial narrative, which was particularly popular among readers. According to Chamberlain, the Cape-to-Cairo plan even carried ‘sentimental’ meaning for the British public, just as Samoa did for the German one. It is clear that Rhodes and the journalists were both complicit in perpetuating the colonial agenda, using the popular sentiment to further their own interests.[CITATION-95]"
    assert get_context(text) == expected_context

    # Test case 4: With [MASK], no succeeding context
    text = "be a new land’.[MASK] Thus, as with the stories about his early career and resurrection after the raid, there was the theme of Rhodes having overcome incredible difficulties. For the press, this theme of ‘civilising Africa’ fitted within the broader colonial narrative, which was particularly popular among readers. According to Chamberlain, the Cape-to-Cairo plan even carried ‘sentimental’ meaning for the British public, just as Samoa did for the German one. It is clear that Rhodes and the journalists were both complicit in perpetuating the colonial agenda, using the popular sentiment to further their own interests.[CITATION-95]"
    expected_context = "Thus, as with the stories about his early career and resurrection after the raid, there was the theme of Rhodes having overcome incredible difficulties. For the press, this theme of ‘civilising Africa’ fitted within the broader colonial narrative, which was particularly popular among readers. According to Chamberlain, the Cape-to-Cairo plan even carried ‘sentimental’ meaning for the British public, just as Samoa did for the German one. It is clear that Rhodes and the journalists were both complicit in perpetuating the colonial agenda, using the popular sentiment to further their own interests.[CITATION-95]"
    assert get_context(text) == expected_context
