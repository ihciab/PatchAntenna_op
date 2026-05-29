import os
import shutil
import sys
import time
from pathlib import Path
import cst
from cst.interface import DesignEnvironment
import cst.results

join_break = '\n'


def _prepend_to_path(path: Path) -> None:
    resolved = str(path.resolve())
    current = os.environ.get("PATH", "")
    entries = current.split(os.pathsep) if current else []
    if resolved not in entries:
        os.environ["PATH"] = resolved + os.pathsep + current if current else resolved


def _ensure_cst_runtime_environment() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    compat_dir = repo_root / "tools"
    if (compat_dir / "wmic.cmd").exists() and shutil.which("wmic") is None:
        _prepend_to_path(compat_dir)
        os.environ.setdefault("WMIC_COMPAT_PYTHON", sys.executable)

    cst_runtime_dir = Path(cst.__file__).resolve().parents[2]
    _prepend_to_path(cst_runtime_dir)

def cst_auto_init(modeler, fmin, fmax):
    #给出基本的仿真需求
    modeler.add_to_history('set frequency', cst_set_frequency(fmin, fmax))
    modeler.add_to_history('set planewave', cst_set_planewave())
    modeler.add_to_history('set boundaries', cst_set_boundaries())
    modeler.add_to_history('set background', cst_set_background())
    modeler.add_to_history('set change solver', cst_change_solver())
    modeler.add_to_history('set timesolver', cst_set_solver2f())
    # The volume field monitor is not needed for S-parameter extraction and
    # can increase memory pressure on low-memory machines.

def cst_change_solver():
    cst_command = 'ChangeSolverType "HF Frequency Domain"'
    return cst_command

def cst_close_project(de:cst.interface.DesignEnvironment, proj:cst.interface.Project, save_flag:bool=True) -> None:
    if save_flag:
        proj.save()
    proj.close()
    de.close()
    return proj

def cst_create_brick(name, x1:float, x2:float, y1:float, y2:float, z:float, component='component1', material='PEC'):
    cst_command = ['With Brick',
                   '.Reset ',
                   f'.Name "{name}"',
                   f'.Component "{component}"',
                   f'.Material "{material}"',
                   f'.Xrange "{x1}", "{x2}"',
                   f'.Yrange "{y1}", "{y2}"',
                   f'.Zrange "{z}", "{z}"',
                   '.Create',
                   'End With',
                   ]
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_create_project(s_path:str, save_flag:bool=True, result:bool=True) -> None:
    #判断是否存在工程，如果存在就不创建，否则就创建
    _ensure_cst_runtime_environment()
    project_path = Path(s_path).expanduser().resolve()
    if project_path.exists():
        pass
    else:
        project_path.parent.mkdir(parents=True, exist_ok=True)
        de = DesignEnvironment.new()
        proj = de.new_mws()
        if save_flag:
            try:
                proj.save(path=str(project_path), include_results=result)
                for _ in range(20):
                    if project_path.exists():
                        break
                    time.sleep(0.2)
            finally:
                proj.close()
                de.close()

def cst_create_material(m_type:str,
                        m_name:str,
                        m_folder:str='',
                        rho:float=0.0,
                        para1:float=0.0,
                        para2:float=0.0,
                        color=None,
                        transparent:float=0.0):
    if color is None:
        color = [0, 1, 1]
    cst_command = None
    if m_type == 'Normal':
        cst_command = ['With Material',
                       '.Reset',
                       f'.Type "{m_type}"',
                       f'.Name "{m_name}"',
                       f'.Folder "{m_folder}"',
                       f'.Rho "{rho}"',
                       f'.Epsilon "{para1}"',
                       f'.TanD "{para2}"',
                       f'.Colour "{color[0]}", "{color[1]}", "{color[2]}"',
                       f'.Transparency "{transparent}" ',
                       '.Create',
                       'End With',
                    ]
    elif m_type == 'Lossy metal':
        # print('1')
        cst_command = ['With Material',
                       '.Reset',
                       f'.Type "{m_type}"',
                       f'.Name "{m_name}"',
                       f'.Folder "{m_folder}"',
                       f'.Rho "{rho}"',
                       f'.Sigma "{para1}"',
                       f'.Mu "{para2}"',
                       f'.Colour "{color[0]}", "{color[1]}", "{color[2]}"',
                       f'.Transparency "{transparent}" ',
                       '.Create',
                       'End With',
                    ]
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_curves(name:str, curve:str, contour):
    if len(contour) < 2:
        return ValueError('曲线有误请检查')
    cst_command = ['With Polygon',
                   '.Reset',
                   f'.Name "{name}"',
                   f'.Curve "{curve}"',
                   '.Create',
                   'End With',]
    for node in contour:
        cst_command.insert(-2, f'.LineTo "{node[0]}", "{node[1]}"')
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_define_floquetport(port_min:int=1, port_max:int=1):
    cst_command = ['With FloquetPort',
                   '.Reset',
                   '.SetDialogTheta "0"',
                   '.SetDialogPhi "0"',
                   '.SetPolarizationIndependentOfScanAnglePhi "0.0", "False"',
                   '.SetSortCode "+beta/pw"',
                   '.SetCustomizedListFlag "False"',
                   '.Port "Zmin"',
                   f'.SetNumberOfModesConsidered "{port_min}"',
                   '.SetDistanceToReferencePlane "-25"',
                   '.SetUseCircularPolarization "False"',
                   '.Port "Zmax"',
                   f'.SetNumberOfModesConsidered "{port_max}"',
                   '.SetDistanceToReferencePlane "-25"',
                   '.SetUseCircularPolarization "False"',
                   'End With',
                   ]
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_del_curves(curve):
    cst_command = [f'Curve.DeleteCurve "{curve}"']
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_del_subtract(component1, solid1, component2, solid2):
    cst_command = [f'Solid.Subtract "{component1}:{solid1}", "{component2}:{solid2}"']
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_del_solid(component, name):
    cst_command = [f'Solid.Delete "{component}:{name}"']
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_excitation2signal(name:str, f_name:str, change:bool=True):
    cst_command = ['With TimeSignal',
                   '.Reset',
                   f'.Name "{name}"',
                   '.SignalType "Import"',
                   '.ProblemType "High Frequency"',
                   f'.FileName "{f_name}"',
                   # '.Id "2"',
                   '.UseCopyOnly "true"',
                   '.Periodic "False"',
                   '.Create',
                   'End With']
    if change:
        cst_command.insert(-1, f'.ExcitationSignalAsReference "{name}", "High Frequency"')

    cst_command = join_break.join(cst_command)
    return cst_command

def cst_excitation2use(name:str):
    cst_command = ['With TimeSignal',
                   f'.ExcitationSignalAsReference "{name}", "High Frequency"',
                   'End With'
                   ]
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_export_pic(result_path, name='FSS'):
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)
    cst_command = ['Plot.RestoreView "Perspective"',
                   'Plot.DrawWorkplane "False"',
                   'Plot.ZoomTostructure',
                   'Plot.ZoomToStructure',
                   'Plot.ZoomTostructure',
                   'Plot.ZoomToStructure',
                   f'Plot.ExportImage("{result_path}\\{name}.png",1024,768)'
                   ]
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_extrudecurve(name:str, curve:str,
                     component:str='component',
                     material:str='PEC',
                     thickness:float=0,
                     twistangle:float=0,
                     taperangle:float=0):
    cst_command = ['With ExtrudeCurve',
                   '.Reset',
                   f'.Name "{name}"',
                   f'.Component "{component}"',
                   f'.Material "{material}"',
                   f'.Thickness "{thickness}"',
                   f'.Twistangle "{twistangle}"',
                   f'.Taperangle "{taperangle}"',
                   '.DeleteProfile "False"',
                   f'.Curve "{curve}"',
                   '.Create',
                   'End With',]
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_heal_all_shapes():
    cst_command = ['With Healing',
                   '.Reset',
                   '.HealAll',
                   'End With',
                   ]
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_import_stl(stl_path:str, stl_name:str, component='default', ):
    cst_command = ['With STL',
                   '.Reset',
                   f'.FileName ("{stl_path}")',
                   f'.Name("{stl_name}")',
                   f'.Component("{component}")',
                   f'.ImportToActiveCoordinateSystem(False)',
                   f'.ScaleToUnit "False"',
                   f'.ImportFileUnits "m"',
                   '.Read',
                   'End With',
                ]
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_load_material(material):
    cst_command = []
    if material == 'FR4':
        cst_command = [
            'With Material',
            '.Reset',
            '.Name "FR4"',
            '.Folder ""',
            '.FrqType "all"',
            '.Type "Normal"',
            '.SetMaterialUnit "GHz", "mm"',
            '.Epsilon "4.3"',
            '.Mu "1.0"',
            '.Kappa "0.0"',
            '.TanD "0.025"',
            '.TanDFreq "10.0"',
            '.TanDGiven "True"',
            '.TanDModel "ConstTanD"',
            '.KappaM "0.0"',
            '.TanDM "0.0"',
            '.TanDMFreq "0.0"',
            '.TanDMGiven "False"',
            '.TanDMModel "ConstKappa"',
            '.DispModelEps "None"',
            '.DispModelMu "None"',
            '.DispersiveFittingSchemeEps "General 1st"',
            '.DispersiveFittingSchemeMu "General 1st"',
            '.UseGeneralDispersionEps "False"',
            '.UseGeneralDispersionMu "False"',
            '.Rho "0.0"',
            '.ThermalType "Normal"',
            '.ThermalConductivity "0.3"',
            '.SetActiveMaterial "all"',
            '.Colour "0.94", "0.82", "0.76"',
            '.Wireframe "False"',
            '.Transparency "0"',
            '.Create',
            'End With',
        ]
    elif material == 'Rogers RT-duroid 5880 (loss free)':
        print('RT5880')
        cst_command = [
            'With Material',
            '.Reset',
            '.Name "Rogers RT-duroid 5880 (loss free)"',
            '.Folder ""',
            '.FrqType "all"',
            '.Type "Normal"',
            '.SetMaterialUnit "GHz", "mm"',
            '.Epsilon "2.2"',
            '.Mu "1.0"',
            '.Kappa "0.0"',
            '.TanD "0.0"',
            '.TanDFreq "0.0"',
            '.TanDGiven "False"',
            '.TanDModel "ConstTanD"',
            '.KappaM "0.0"',
            '.TanDM "0.0"',
            '.TanDMFreq "0.0"',
            '.TanDMGiven "False"',
            '.TanDMModel "ConstKappa"',
            '.DispModelEps "None"',
            '.DispModelMu "None"',
            '.DispersiveFittingSchemeEps "General 1st"',
            '.DispersiveFittingSchemeMu "General 1st"',
            '.UseGeneralDispersionEps "False"',
            '.UseGeneralDispersionMu "False"',
            '.Rho "0.0"',
            '.ThermalType "Normal"',
            '.ThermalConductivity "0.20"',
            '.SetActiveMaterial "all"',
            '.Colour "0.75", "0.95", "0.85"',
            '.Wireframe "False"',
            '.Transparency "0"',
            '.Create',
            'End With',
        ]
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_open_project(cst_file:str) -> cst.interface.Project:
    _ensure_cst_runtime_environment()
    cst_path = Path(cst_file).expanduser().resolve()
    if not cst_path.exists():
        raise FileNotFoundError(f"CST project file not found: {cst_path}")

    last_error = None
    for _ in range(5):
        de = DesignEnvironment.new()
        try:
            time.sleep(0.5)
            print(cst_path)
            proj = de.open_project(str(cst_path))
            result = cst.results.ProjectFile(str(cst_path), allow_interactive=True)
            return de, proj, result
        except RuntimeError as exc:
            last_error = exc
            try:
                de.close()
            except Exception:
                pass
            time.sleep(1.0)

    raise RuntimeError(
        f"Failed to open CST project after 5 attempts: {cst_path}. Last error: {last_error}"
    )

def cst_scale(name, scale, rep:int=1):
    cst_command = ['With Transform',
                   '.Reset',
                   f'.Name "{name}"',
                   '.Origin "Free"',
                   '.Center "0", "0", "0"',
                   f'.ScaleFactor "{scale}", "{scale}", "1"',
                   '.MultipleObjects "False"',
                   '.GroupObjects "False"',
                   f'.Repetitions "{rep}"',
                   '.MultipleSelection "False"',
                   # '.AutoDestination "True"', # 2022版本不行
                   '.Transform "Shape", "Scale"',
                   'End With',
                   ]
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_set_background():
    cst_command = ['With Background',
                   '.ResetBackground',
                   '.XminSpace "0.0"',
                   '.XmaxSpace "0.0"',
                   '.YminSpace "0.0"',
                   '.YmaxSpace "0.0"',
                   '.ZminSpace "25"',
                   '.ZmaxSpace "25"',
                   '.ApplyInAllDirections "False"',
                   'End With',
                   ]
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_set_boundaries():
    cst_command = ['With Boundary',
                   '.Xmin "expanded open"',
                   '.Xmax "expanded open"',
                   '.Ymin "expanded open"',
                   '.Ymax "expanded open"',
                   '.Zmin "expanded open"',
                   '.Zmax "expanded open"',
                   '.Xsymmetry "none"',
                   '.Ysymmetry "none"',
                   '.Zsymmetry "none"',
                   '.ApplyInAllDirections "True"',
                   '.XPeriodicShift "0.0"',
                   '.YPeriodicShift "0.0"',
                   '.ZPeriodicShift "0.0"',
                   '.PeriodicUseConstantAngles "False"',
                   'End With',
    ]
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_set_frequency(fmin:float=0.0, fmax:float=1.0):
    cst_command = f'Solver.FrequencyRange "{fmin}", "{fmax}"'
    return cst_command

def cst_set_material(m_folder:str='default:solid1', material:str='PEC'):
    cst_command = f'Solid.ChangeMaterial "{m_folder}", "{material}"'
    return cst_command

def cst_set_monitor2filed(f):
    cst_command = ['With Monitor',
                   '.Reset',
                   f'.Name "e-field (f={f})"',
                   '.Dimension "Volume"',
                   '.Domain "Frequency"',
                   '.FieldType "Efield"',
                   f'.MonitorValue "{f}"',
                   '.UseSubvolume "False"',
                   '.Coordinates "Structure"',
                   '.Create',
                   'End With',
                   ]
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_solid_rename(component, old_name, new_name):
    cst_command = [f'Solid.Rename "{component}:{old_name}","{new_name}"']
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_set_planewave(normal:list=None, e_vector:list=None):
    if normal is None:
        normal = [0, 0, 1]
    if e_vector is None:
        e_vector = [1, 0, 0]
    cst_command = ['With PlaneWave',
                   '.Reset',
                   f'.Normal "{normal[0]}", "{normal[1]}", "{normal[2]}"',
                   f'.EVector "{e_vector[0]}", "{e_vector[1]}", "{e_vector[2]}"',
                   '.Polarization "Linear"',
                   '.Store',
                   'End With'
    ]
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_set_solver2f():
    cst_command = ['With FDSolver',
                   '.Reset',
                   '.SetMethod "Tetrahedral", "General purpose"',
                   '.OrderTet "Second"',
                   '.OrderSrf "First"',
                   '.MeshAdaptionTet "False"',
                   '.Stimulation "1", "All"',
                   'End With',
                    ]
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_spline_curves(name:str, curve:str, contour):
    if len(contour) < 2:
        return ValueError('曲线有误请检查')
    cst_command = ['With Spline',
                   '.Reset',
                   f'.Name "{name}"',
                   f'.Curve "{curve}"',
                   f'.SetInterpolationType "PointInterpolation" ',
                   '.Create',
                   'End With',]
    for node in contour:
        cst_command.insert(-2, f'.LineTo "{node[0]}", "{node[1]}"')
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_straight_line(name:str, curve:str, node1, node2):
    start_x, start_y = node1
    end_x, end_y = node2
    cst_command = ['With Polygon',
                   '.Reset',
                   f'.Name "{name}"',
                   f'.Curve "{curve}"',
                   f'.Point "{start_x}", "{start_y}"',
                   f'.LineTo "{end_x}", "{end_y}"',
                   '.Create',
                   'End With',]
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_translate(name, x=0, y=0, z=0, copy_flag:bool=False, repetitions:int=1):
    cst_command = ['With Transform',
                   '.Reset',
                   f'.Name "{name}"',
                   f'.Vector "{x}", "{y}", "{z}"',
                   '.UsePickedPoints "False"',
                   '.InvertPickedPoints "False"',
                   f'.MultipleObjects "{copy_flag}"',
                   '.GroupObjects "False"',
                   f'.Repetitions "{repetitions}"',
                   '.MultipleSelection "False"',
                   # '.AutoDestination "True"',
                   '.Transform "Shape", "Translate"',
                   'End With',
                   ]
    cst_command = join_break.join(cst_command)
    return cst_command

def cst_waveguide_port(orientation, x1, x2, y1, y2, z1, z2):
    if orientation in {"x", "y", "z"}:
        orientation = f"{orientation}max"
    cst_command = ['With Port',
                   '.Reset',
                   '.PortNumber "1"',
                   '.Label ""',
                   '.Folder ""',
                   '.NumberOfModes "1"',
                   '.AdjustPolarization "False"',
                   '.PolarizationAngle "0.0"',
                   '.ReferencePlaneDistance "0"',
                   '.TextSize "50"',
                   '.TextMaxLimit "0"',
                   '.Coordinates "Free"',
                   f'.Orientation "{orientation}"',
                   '.PortOnBound "False"',
                   '.ClipPickedPortToBound "False"',
                   f'.Xrange "{x1}", "{x2}"',
                   f'.Yrange "{y1}", "{y2}"',
                   f'.Zrange "{z1}", "{z2}"',
                   '.XrangeAdd "0.0", "0.0"',
                   '.YrangeAdd "0.0", "0.0"',
                   '.ZrangeAdd "0.0", "0.0"',
                   '.SingleEnded "False"',
                   '.WaveguideMonitor "False"',
                   '.Create',
                   'End With',]
    cst_command = join_break.join(cst_command)
    return cst_command




