"""ParaView pipeline for the 3D snap surface (CURRENTv4, 0.02 run).

Input: data/snap_surface_3d/snap_mac_indicator.vtu from make_mac_vtu.py (points are (phi1, phi2, d)).
Surface = Contour of the nodal neighbour-MAC indicator at 1 - MAC = 0.1 (MAC = 0.90),
then Laplacian smoothing and normals. Run:
    pvpython --force-offscreen-rendering render_snap.py
"""
import os
from paraview.simple import *

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
INPUT = os.path.join(REPO, "data", "snap_surface_3d", "snap_mac_indicator.vtu")
FIGURES = os.path.join(REPO, "presentation", "figures")
STATE = os.path.join(REPO, "data", "snap_surface_3d", "snap_surface.pvsm")
os.makedirs(FIGURES, exist_ok=True)

ZS = 8.0                  # vertical exaggeration: d spans 0.19, phi spans 1.6
ISO = 0.1                 # 1 - MAC threshold
FOCAL = [0, 0, 0.895 * ZS]
INK, GREY = [0.086, 0.133, 0.18], [0.54, 0.59, 0.64]
D_RGB = [0.80, 0.086, 0.133, 0.18, 0.85, 0.18, 0.35, 0.42, 0.895, 0.165, 0.616, 0.561,
         0.94, 0.914, 0.769, 0.416, 0.99, 0.851, 0.51, 0.169]
F_RGB = [0, 0.96, 0.97, 0.98, 20, 0.62, 0.80, 0.80, 45, 0.165, 0.45, 0.50, 70, 0.086, 0.133, 0.18]

data = XMLUnstructuredGridReader(FileName=[INPUT])
iso = Contour(Input=data, ContourBy=['POINTS', 'mac_jump'], Isosurfaces=[ISO])
smooth = Smooth(Input=iso); smooth.NumberofIterations = 400
snap = SurfaceNormals(Input=smooth)


def new_view(size=(1800, 1350)):
    v = CreateView('RenderView'); v.ViewSize = list(size); v.OrientationAxesVisibility = 0
    v.UseColorPaletteForBackground = 0; v.BackgroundColorMode = 'Single Color'; v.Background = [1, 1, 1]
    SetActiveView(v); return v


def lut(name, rgb):
    l = GetColorTransferFunction(name); l.RGBPoints = rgb; l.ColorSpace = 'RGB'; return l


def colour_bar(l, v, title, pos=(0.89, 0.25), ticks=None):
    b = GetScalarBar(l, v); b.Title = title; b.ComponentTitle = ''
    b.TitleColor = INK; b.LabelColor = INK; b.TitleFontSize = 44; b.LabelFontSize = 36; b.ScalarBarThickness = 28
    b.WindowLocation = 'Any Location'; b.Position = list(pos); b.ScalarBarLength = 0.6
    b.RangeLabelFormat = '{:.2f}'; b.LabelFormat = '{:.2f}'; b.AddRangeLabels = 0; b.Visibility = 1
    if ticks:
        b.UseCustomLabels = 1; b.CustomLabels = list(ticks)


def frame(v, z_label=True):
    o = Show(Outline(Input=data), v); o.Scale = [1, 1, ZS]
    o.AmbientColor = o.DiffuseColor = GREY; o.LineWidth = 1.5
    g = o.DataAxesGrid; g.GridAxesVisibility = 1; g.ShowGrid = 0; g.GridColor = GREY
    g.XTitle, g.YTitle, g.ZTitle = 'φ₁', 'φ₂', ('d' if z_label else '')
    for ax in 'XYZ':
        setattr(g, f'{ax}TitleColor', INK); setattr(g, f'{ax}LabelColor', INK)
        setattr(g, f'{ax}TitleFontSize', 52); setattr(g, f'{ax}LabelFontSize', 38)
        setattr(g, f'{ax}AxisUseCustomLabels', 1)
    g.XAxisLabels = g.YAxisLabels = [-0.4, 0, 0.4]; g.ZAxisLabels = []


def show_snap(v, colour='d', opacity=1.0, solid=None):
    s = Show(snap, v); s.Scale = [1, 1, ZS]; s.Specular = 0.3; s.Opacity = opacity
    if solid:
        ColorBy(s, None); s.AmbientColor = s.DiffuseColor = solid
    else:
        ColorBy(s, ('POINTS', colour)); s.LookupTable = lut(colour, D_RGB)
    return s


def camera(v, direction, up=(0, 0, 1), dist=6.2, parallel=False, scale=1.0, shift=(0, 0, 0)):
    n = sum(x * x for x in direction) ** 0.5
    focal = [FOCAL[i] + shift[i] for i in range(3)]
    v.CameraFocalPoint = focal; v.CameraViewUp = list(up)
    v.CameraPosition = [focal[i] + direction[i] / n * dist for i in range(3)]
    v.CameraViewAngle = 30; v.CameraParallelProjection = int(parallel)
    if parallel:
        v.CameraParallelScale = scale


def save(v, name):
    SaveScreenshot(os.path.join(FIGURES, f'{name}.png'), v, ImageResolution=v.ViewSize, TransparentBackground=0)


# 1. Oblique view, coloured by d
v = new_view(); show_snap(v); frame(v)
colour_bar(GetColorTransferFunction('d'), v, 'd', pos=(0.86, 0.2), ticks=(0.8, 0.85, 0.9, 0.95, 0.99))
camera(v, (3.2, -4.2, 3.0), dist=5.6)
save(v, 'snap3d_oblique')

# 2. Looking straight down the d axis: nested L-shaped snap loci
v = new_view((1800, 1350)); show_snap(v); frame(v, z_label=False)
colour_bar(GetColorTransferFunction('d'), v, 'd', pos=(0.86, 0.2), ticks=(0.8, 0.85, 0.9, 0.95, 0.99))
camera(v, (0, 0, 1), up=(0, 1, 0), parallel=True, scale=1.0, shift=(0.12, 0, 0))
save(v, 'snap3d_top')

# 3. Cut-away at d = 0.90: surface above the plane, fx on the plane. The surface edge
#    should run exactly along the step in fx (~27 -> ~57).
D_CUT = 0.90
v = new_view(); frame(v)
below = Clip(Input=snap); below.ClipType = 'Plane'; below.ClipType.Origin = [0, 0, D_CUT]
below.ClipType.Normal = [0, 0, 1]; below.Invert = 0
s = Show(below, v); s.Scale = [1, 1, ZS]; s.Specular = 0.3; s.Opacity = 0.5
ColorBy(s, ('POINTS', 'd')); s.LookupTable = lut('d', D_RGB)
sl = Slice(Input=data); sl.SliceType = 'Plane'; sl.SliceType.Origin = [0, 0, D_CUT]; sl.SliceType.Normal = [0, 0, 1]
s = Show(sl, v); s.Scale = [1, 1, ZS]; ColorBy(s, ('POINTS', 'fx'))
flut = lut('fx', F_RGB); s.LookupTable = flut; s.Ambient = 0.35; s.Diffuse = 0.75
flut.AutomaticRescaleRangeMode = 'Never'; flut.RGBPoints = F_RGB      # fixed 0-70 after ColorBy
colour_bar(flut, v, 'fx', pos=(0.86, 0.2), ticks=(0, 10, 20, 30, 40, 50, 60, 70))
camera(v, (2.6, -3.4, 4.2), dist=5.6)
save(v, 'snap3d_slices')

SaveState(STATE)
print('done')
