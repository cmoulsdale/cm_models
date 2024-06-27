from cm_models import util


def test_pseudo_docstring():
    """compare to known pseudo docstring"""

    default_parameters = dict(a=("1", "a"), b=(int, "b"), c=(3.0, "c"))
    docstring = """Parameters
----------

a : str, optional
    a (default is 1)
b : int
    b
c : float, optional
    c (default is 3.0)"""
    assert util.pseudo_docstring(default_parameters) == docstring
