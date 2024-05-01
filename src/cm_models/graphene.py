# from __future__ import annotations

# from collections.abc import Iterable, Iterator

# import numpy as np
# import math

# from . import abstracts, msl, het


# class Graphene(abstracts.Model2D):
#     """Base class for graphene"""

#     extra_model_parameters = dict(a=(0.246, "Lattice constant. [nm]"))

#     extra_function_parameters = dict(K_plus=(False, "In K+ valley."))

#     @staticmethod
#     def resolve_valley(elements, K_plus):
#         if K_plus:
#             yield from elements
#         else:
#             for i1, i2, value, operators in elements:
#                 yield (
#                     i1,
#                     i2,
#                     (
#                         value
#                         if sum((op == "k" or op == "kc") for op in operators) % 2 == 0
#                         else -value
#                     ),
#                     tuple(
#                         "k" if op == "kc" else "kc" if op == "k" else op
#                         for op in operators
#                     ),
#                 )


# class GrapheneSuperlattice(Graphene, het.ModelSuperlattice2D):
#     """Base class for graphene superlattices"""


# uV_partial_model_parameters = (
#     (
#         "Vp",
#         "Inversion symmetric moire V-potential V+, "
#         "overridden by explicit u-potentials [eV].",
#     ),
#     (
#         "Vm",
#         "Inversion antisymmetric moire V-potential V-, "
#         "overridden by explicit u-potentials [eV].",
#     ),
#     ("u0p", "Inversion symmetric moire u-potential u0+ [eV]."),
#     ("u1p", "Inversion symmetric moire u-potential u1+ [eV]."),
#     ("u3p", "Inversion symmetric moire u-potential u3+ [eV]."),
#     ("u0m", "Inversion antisymmetric moire u-potential u0- [eV]."),
#     ("u1m", "Inversion antisymmetric moire u-potential u1- [eV]."),
#     ("u3m", "Inversion antisymmetric moire u-potential u3- [eV]."),
#     ("tau_x", "x-component of offset at graphene-hBN interface [nm]."),
#     ("tau_y", "y-component of offset at graphene-hBN interface [nm]."),
#     ("inverted", "hBN layer is inverted."),
# )


# class GrapheneHBNSuperlattice(GrapheneSuperlattice, msl.SuperlatticeTriangular2D):
#     """Arbitrary graphene/hBN superlattice"""

#     extra_model_parameters = dict(
#         delta=(0.018, "Graphene/hBN lattice constant mismatch."),
#         theta=(0.0, "hBN twist. [rad]"),
#     )

#     remove_model_parameters_names = ("period", "phi")

#     def resolve_model_parameters(self):
#         delta = self.delta
#         theta = self.theta
#         cos = math.cos(theta)
#         self.period = self.a / math.sqrt(
#             1.0 - 2.0 * cos / (1.0 + delta) + 1.0 / (1.0 + delta) ** 2
#         )
#         self.phi = (
#             math.atan2(1.0 - cos / (1.0 + delta), math.sin(theta) / (1.0 + delta))
#             - 0.5 * math.pi
#         )

#     def fill_namespace(self):
#         # first star of graphene Bragg vectors
#         angle = np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False)
#         self.g_first = (4.0 * math.pi / (math.sqrt(3.0) * self.a)) * np.stack(
#             [-np.sin(angle), np.cos(angle)], axis=1
#         )


# def uV_extra_model_parameters_names(suffix: str = "") -> tuple[str]:
#     """Names of extra uV model parameters of given suffix"""

#     return tuple(f"{name}{suffix}" for name, description in uV_partial_model_parameters)


# def uV_extra_model_parameters(
#     suffix="",
#     id="",
#     Vp=0.017,
#     Vm=0.0,
#     u0p=0.0,
#     u1p=0.0,
#     u3p=0.0,
#     u0m=0.0,
#     u1m=0.0,
#     u3m=0.0,
#     tau_x=0.0,
#     tau_y=0.0,
#     inverted=False,
# ):
#     """Extra uV model parameters of given suffix and ID.

#     See ``model_parameters()``.

#     TODO: add parameters

#     """

#     return dict(
#         (
#             [
#                 f"{name}{suffix}",
#                 (value, f"{description} ({id})" if id else description),
#             ]
#             for value, (name, description) in zip(
#                 [Vp, Vm, u0p, u1p, u3p, u0m, u1m, u3m, tau_x, tau_y, inverted],
#                 uV_partial_model_parameters,
#             )
#         )
#     )


# class GrapheneHBNSuperlatticeUV(GrapheneHBNSuperlattice):
#     """Arbitrary graphene/hBN superlattice with uV parameterisation"""

#     def uV_fill_namespace(self, suffix: str = ""):
#         """Fill namespace with secondary uV parameters of given suffix"""

#         names = uV_extra_model_parameters_names(suffix=suffix)
#         Vp, Vm, u0p, u1p, u3p, u0m, u1m, u3m, tau_x, tau_y, inverted = (
#             getattr(self, name) for name in names
#         )

#         if any([u0p, u1p, u3p, u0m, u1m, u3m]):
#             # explicit u-potentials override V-potentials
#             Vp = None
#             Vm = None
#         else:
#             # V-potentials determine u-potentials
#             u0p = 0.5 * Vp
#             u1p = -Vp
#             u3p = -0.5 * math.sqrt(3.0) * Vp
#             u0m = -0.5 * Vm
#             u1m = -Vm
#             u3m = -0.5 * math.sqrt(3.0) * Vm

#         for name, value in zip(names[:-3], (Vp, Vm, u0p, u1p, u3p, u0m, u1m, u3m)):
#             setattr(self, name, value)

#         phase = np.exp(1j * (tau_x * self.g_first[:, 0] + tau_y * self.g_first[:, 1]))
#         if inverted:
#             setattr(self, f"WAA{suffix}", phase * (u0p - 1j * u0m + 1j * u3p - u3m))
#             setattr(self, f"WBB{suffix}", phase * (u0p - 1j * u0m - 1j * u3p + u3m))
#             setattr(self, f"WBA{suffix}", phase * (u1p + 1j * u1m))
#         else:
#             setattr(self, f"WAA{suffix}", phase * (u0p + 1j * u0m + 1j * u3p + u3m))
#             setattr(self, f"WBB{suffix}", phase * (u0p + 1j * u0m - 1j * u3p - u3m))
#             setattr(self, f"WBA{suffix}", phase * (u1p - 1j * u1m))

#     def uV_super_elements(
#         self, A, B, K_plus: bool, suffix: str = ""
#     ) -> Iterator[Iterable]:
#         """Superlattice matrix elements with secondary uV parameters of given suffix"""

#         for m, WAA, WBB, WBA, phase in zip(
#             range(6),
#             getattr(self, f"WAA{suffix}"),
#             getattr(self, f"WBB{suffix}"),
#             getattr(self, f"WBA{suffix}"),
#             np.exp(
#                 np.linspace(
#                     0j, 2j * math.pi if K_plus else -2j * math.pi, 6, endpoint=False
#                 )
#             ),
#         ):
#             if m % 2 == 0:
#                 # diagonal: u0+, u0-, u3+, u3-
#                 yield (A, A, WAA, (m,))
#                 yield (B, B, WBB, (m,))

#                 # off-diagonal: u1+, u1-
#                 yield (B, A, phase * WBA, (m,))
#             else:
#                 # diagonal: u0+, u0-, u3+, u3-
#                 yield (A, A, WAA.conjugate(), (m,))
#                 yield (B, B, WBB.conjugate(), (m,))

#                 # off-diagonal: u1+, u1-
#                 yield (B, A, -phase * WBA.conjugate(), (m,))
