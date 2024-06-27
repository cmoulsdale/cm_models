"""Bilayer graphene.
"""

import math
import numpy as np

from .graphene import (
    Graphene,
    GrapheneHBNSuperlatticeUV,
    uV_extra_model_parameters_names,
    uV_extra_model_parameters,
)


class BLG_ABC(Graphene):
    """Bilayer graphene abstract base class."""

    extra_model_parameters = dict(
        g0=(3.16, "Intralayer SWMcC coupling. [eV]"),
        g1=(0.381, "Vertical interlayer SWMcC coupling. [eV]"),
        g3=(0.38, "First skew interlayer SWMcC coupling. [eV]"),
        g4=(0.14, "Second skew interlayer SWMcC coupling. [eV]"),
        Delta_U=(0.0, "Interlayer potential asymmetry. [eV]"),
        Delta_AB=(0.0, "Sublattice potential asymmetry. [eV]"),
        Delta_p=(0.022, "Dimer potential asymmetry. [eV]"),
    )

    def fill_namespace(self):
        # velocities
        self.v0 = 0.5 * math.sqrt(3.0) * self.a * self.g0
        self.v3 = 0.5 * math.sqrt(3.0) * self.a * self.g3
        self.v4 = 0.5 * math.sqrt(3.0) * self.a * self.g4


class BLG(BLG_ABC):
    """Bilayer graphene.

    4-band model of bilayer graphene in the (A0, B0, A1, B1) sublattice basis.

    Parameters
    ----------

    a : float, optional
        Graphene lattice parameter [nm]. (default is 0.246)
    g0 : float, optional
        gamma0 coupling [eV]. (default is 3.16)
    g1 : float, optional
        gamma1 coupling [eV]. (default is 0.381)
    g3 : float, optional
        gamma3 coupling [eV]. (default is 0.38)
    g4 : float, optional
        gamma4 coupling [eV]. (default is 0.14)
    Delta_AB : float, optional
        External sublattice asymmetry [eV]. (default is 0.0)
    Delta_U : float, optional
        External interlayer asymmetry [eV]. (default is 0.0)
    Delta_p : float, optional
        Dimer asymmetry [eV]. (default is 0.022)

    """

    def sublattices(self, **model_kwargs):
        return ("A0", "B0", "A1", "B1")

    def extra_major_elements(self, *, K_plus, **model_kwargs):
        # gamma1
        yield ("A1", "B0", self.g1, ())

        yield from self.resolve_valley(
            [
                # gamma0
                ("B0", "A0", self.v0, ("k",)),
                ("B1", "A1", self.v0, ("k",)),
            ],
            K_plus,
        )

    def extra_minor_elements(self, *, K_plus, **model_kwargs):
        # on site potentials
        yield ("A0", "A0", 0.5 * (self.Delta_AB - self.Delta_U), ())
        yield (
            "B0",
            "B0",
            0.5 * (-self.Delta_AB - self.Delta_U) + self.Delta_p,
            (),
        )
        yield (
            "A1",
            "A1",
            0.5 * (self.Delta_AB + self.Delta_U) + self.Delta_p,
            (),
        )
        yield ("B1", "B1", 0.5 * (-self.Delta_AB + self.Delta_U), ())

        yield from self.resolve_valley(
            [
                # gamma3
                ("B1", "A0", self.v3, ("kc",)),
                # gamma4
                ("A1", "A0", -self.v4, ("k",)),
                ("B1", "B0", -self.v4, ("k",)),
            ],
            K_plus,
        )


class BLG2(BLG_ABC):
    def sublattices(self, **model_kwargs):
        return ("A0", "B1")

    def fill_namespace(self):
        # mass
        self.m = self.g1 / (2.0 * self.v0**2)

        # particle-hole symmetry
        self.alpha = 2.0 * self.v4 / self.v0 + self.Delta_p / self.g1

    def extra_major_elements(self, *, K_plus, **model_kwargs):
        # mass
        yield from self.resolve_valley(
            [("B1", "A0", -1.0 / (2.0 * self.m), ("k", "k"))], K_plus
        )

    def extra_minor_elements(self, *, K_plus, **model_kwargs):
        # on site potentials
        yield ("A0", "A0", -0.5 * self.Delta_U, ())
        yield ("B1", "B1", 0.5 * self.Delta_U, ())

        yield from self.resolve_valley(
            [
                # gamma3
                ("B1", "A0", self.v3, ("kc",)),
                # particle-hole and inversion symmetry breaking mass
                (
                    "A0",
                    "A0",
                    (self.alpha + self.Delta_U / self.g1) / (2.0 * self.m),
                    ("kc", "k"),
                ),
                (
                    "B1",
                    "B1",
                    (self.alpha - self.Delta_U / self.g1) / (2.0 * self.m),
                    ("k", "kc"),
                ),
            ],
            K_plus,
        )


class BLGHBN(BLG, GrapheneHBNSuperlatticeUV):
    # default model parameters
    extra_model_parameters = dict(
        **uV_extra_model_parameters("_b", "bottom"),
        **uV_extra_model_parameters("_t", "top"),
        tau_x=(0.0, ""),
        tau_y=(0.0, ""),
    )

    def fill_namespace(self):
        self.uV_fill_namespace("_b")
        self.uV_fill_namespace("_t")

    def extra_super_elements(self, *, K_plus, **model_kwargs):
        yield from self.uV_super_elements("A0", "B0", K_plus, "_b")
        yield from self.uV_super_elements("A1", "B1", K_plus, "_t")


class BLGOnHBN(BLGHBN):
    # default model parameters
    extra_model_parameters = dict(
        [
            *(
                [name, value_and_help]
                for name, value_and_help in uV_extra_model_parameters().items()
                if name not in ("tau_x", "tau_y")
            ),
            ["top_layer", (False, "hBN layer is on top.")],
        ]
    )

    # model parameter keys to pop
    remove_model_parameters_names = (
        *uV_extra_model_parameters_names("_b"),
        *uV_extra_model_parameters_names("_t"),
        "tau_x",
        "tau_y",
    )

    def resolve_model_parameters(self):
        self.tau_x_b = 0.0
        self.tau_y_b = 0.0
        self.tau_x_t = 0.0
        self.tau_y_t = 0.0
        if self.top_layer:
            for name in uV_extra_model_parameters_names()[:-3]:
                setattr(self, f"{name}_b", 0.0)
                setattr(self, f"{name}_t", getattr(self, name))
            self.inverted_b = False
            self.inverted_t = self.inverted
        else:
            for name in uV_extra_model_parameters_names()[:-3]:
                setattr(self, f"{name}_b", getattr(self, name))
                setattr(self, f"{name}_t", 0.0)
            self.inverted_b = self.inverted
            self.inverted_t = False


class BLG2HBN(GrapheneHBNSuperlatticeUV, BLG2):
    # default model parameters
    extra_model_parameters = dict(
        **uV_extra_model_parameters("_b", "bottom"),
        **uV_extra_model_parameters("_t", "top"),
        tau_x=(0.0, ""),
        tau_y=(0.0, ""),
    )

    def fill_namespace(self):
        self.uV_fill_namespace("_b")
        self.uV_fill_namespace("_t")

    def extra_super_elements(self, *, K_plus, **model_kwargs):
        if any(
            [self.u0p_t, self.u0m_t, self.u1p_t, self.u1m_t, self.u3p_t, self.u3m_t]
        ):
            raise NotImplementedError("top hBN layer not implemented")

        for m, WAA_b, WBB_b, WBA_b, phase in zip(
            range(6),
            self.WAA_b,
            self.WBB_b,
            self.WBA_b,
            np.exp(
                np.linspace(
                    0.0j, 2.0j * np.pi if K_plus else -2.0j * np.pi, 6, endpoint=False
                )
            ),
        ):
            if m % 2 == 0:
                # diagonal: u0+, u0-, u3+, u3-
                yield ("A0", "A0", WAA_b, (m,))

                yield from self.resolve_valley(
                    [
                        (
                            "B1",
                            "B1",
                            (self.v0 / self.g1) ** 2 * WBB_b,
                            ("k", m, "kc"),
                        ),
                        # off-diagonal: u1+, u1-
                        (
                            "B1",
                            "A0",
                            -phase * (self.v0 / self.g1) * WBA_b,
                            ("k", m),
                        ),
                    ],
                    K_plus,
                )
            else:
                # diagonal: u0+, u0-, u3+, u3-
                yield ("A0", "A0", WAA_b.conjugate(), (m,))

                yield from self.resolve_valley(
                    [
                        (
                            "B1",
                            "B1",
                            (self.v0 / self.g1) ** 2 * WBB_b.conjugate(),
                            ("k", m, "kc"),
                        ),
                        # off-diagonal: u1+, u1-
                        (
                            "B1",
                            "A0",
                            phase * (self.v0 / self.g1) * WBA_b.conjugate(),
                            ("k", m),
                        ),
                    ],
                    K_plus,
                )


class BLG2OnHBN(BLG2HBN):
    # default model parameters
    extra_model_parameters = dict(
        [
            *(
                [name, value_and_help]
                for name, value_and_help in uV_extra_model_parameters().items()
                if name not in ("tau_x", "tau_y")
            ),
            ["top_layer", (False, "hBN layer is on top.")],
        ]
    )

    # model parameter keys to pop
    remove_model_parameters_names = (
        *uV_extra_model_parameters_names("_b"),
        *uV_extra_model_parameters_names("_t"),
        "tau_x",
        "tau_y",
    )

    def resolve_model_parameters(self):
        self.tau_x_b = 0.0
        self.tau_y_b = 0.0
        self.tau_x_t = 0.0
        self.tau_y_t = 0.0
        if self.top_layer:
            for name in uV_extra_model_parameters_names()[:-3]:
                setattr(self, f"{name}_b", 0.0)
                setattr(self, f"{name}_t", getattr(self, name))
            self.inverted_b = False
            self.inverted_t = self.inverted
        else:
            for name in uV_extra_model_parameters_names()[:-3]:
                setattr(self, f"{name}_b", getattr(self, name))
                setattr(self, f"{name}_t", 0.0)
            self.inverted_b = self.inverted
            self.inverted_t = False
