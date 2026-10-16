from ecalls.paper.float_check import first_page, floats, mention, verdict


def test_floats_reads_kind_number_and_page():
    aux = (r"\newlabel{fig:recall}{{2}{4}{Recall}{figure.2}{}}"
           r"\newlabel{sec:intro}{{I}{1}{Introduction}{section.1}{}}"
           r"\newlabel{tab:premia}{{V}{7}{Premium}{table.5}{}}")
    assert floats(aux) == [("fig", "2", 4), ("tab", "V", 7)]


def test_figure_caption_and_longer_numbers_are_not_mentions():
    p = mention("fig", "3")
    assert p.search("as shown in (Fig. 3a), the premium")
    assert p.search("in Fig. 3.")  # end of a sentence
    assert not p.search("Fig. 3. Look-ahead through the encoder")
    assert not p.search("see Fig. 30")


def test_table_caption_and_longer_numerals_are_not_mentions():
    assert mention("tab", "IV").search("(Table IV, Panel B)")
    assert not mention("tab", "IV").search("TABLE IV Call text")
    assert not mention("tab", "I").search("see Table II")
    assert not mention("tab", "I").search("see Table IV")


def test_first_page_is_one_based_and_none_when_absent():
    pages = ["no mention", "TABLE III caption", "see Table III here", "Table III again"]
    assert first_page(pages, mention("tab", "III")) == 3
    assert first_page(pages, mention("tab", "V")) is None


def test_verdict_flags_each_case():
    assert verdict(4, 4, 6) == ""
    assert verdict(5, 4, 6) == ""
    assert verdict(3, 4, 6) == "before its mention"
    assert verdict(6, 4, 6) == "more than a page after its mention"
    assert verdict(7, 6, 6) == "after the references"
    assert verdict(3, None, 6) == "not mentioned"
