# """
# Convex Hull Analysis for Limit-Average Checking
# """
#
# from typing import List, Tuple, Dict, Any
# import math
#
#
# class ConvexHullChecker:
#     """Convex hull computation and analysis for limit-average constraints."""
#
#     def compute_convex_hull(self, points: List[Tuple[float, float]]) -> Dict[str, Any]:
#         print(f"📐 Computing Convex Hull")
#         print(f"   Input points: {len(points)}")
#
#         if len(points) < 3:
#             return {
#                 'hull_points': points,
#                 'area': 0.0,
#                 'perimeter': self._compute_perimeter(points),
#                 'is_valid': len(points) >= 3
#             }
#
#         # Find convex hull using Graham scan
#         hull_points = self._graham_scan(points)
#
#         area = self._compute_area(hull_points)
#         perimeter = self._compute_perimeter(hull_points)
#
#         result = {
#             'hull_points': hull_points,
#             'area': area,
#             'perimeter': perimeter,
#             'is_valid': True,
#             'original_points': len(points),
#             'hull_size': len(hull_points)
#         }
#
#         print(f"   Hull points: {len(hull_points)}")
#         print(f"   Area: {area:.3f}")
#         print(f"   Perimeter: {perimeter:.3f}")
#
#         return result
#
#     def _graham_scan(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
#         """Graham scan algorithm for convex hull."""
#
#         def cross_product(o, a, b):
#             return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
#
#         points = sorted(set(points))
#         if len(points) <= 1:
#             return points
#
#         # Build lower hull
#         lower = []
#         for p in points:
#             while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
#                 lower.pop()
#             lower.append(p)
#
#         # Build upper hull
#         upper = []
#         for p in reversed(points):
#             while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
#                 upper.pop()
#             upper.append(p)
#
#         # Remove last point of each half because it's repeated
#         return lower[:-1] + upper[:-1]
#
#     def _compute_area(self, points: List[Tuple[float, float]]) -> float:
#         """Compute area of polygon using shoelace formula."""
#         if len(points) < 3:
#             return 0.0
#
#         area = 0.0
#         n = len(points)
#         for i in range(n):
#             j = (i + 1) % n
#             area += points[i][0] * points[j][1]
#             area -= points[j][0] * points[i][1]
#         return abs(area) / 2.0
#
#     def _compute_perimeter(self, points: List[Tuple[float, float]]) -> float:
#         """Compute perimeter of polygon."""
#         if len(points) < 2:
#             return 0.0
#
#         perimeter = 0.0
#         n = len(points)
#         for i in range(n):
#             j = (i + 1) % n
#             dx = points[j][0] - points[i][0]
#             dy = points[j][1] - points[i][1]
#             perimeter += math.sqrt(dx * dx + dy * dy)
#
#         return perimeter
#
#     def point_in_convex_hull(self, point: Tuple[float, float],
#                              hull_points: List[Tuple[float, float]]) -> bool:
#         """Check if a point is inside the convex hull."""
#         if len(hull_points) < 3:
#             return point in hull_points
#
#         def cross_product(o, a, b):
#             return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
#
#         n = len(hull_points)
#         for i in range(n):
#             j = (i + 1) % n
#             if cross_product(hull_points[i], hull_points[j], point) < 0:
#                 return False
#
#         return True
