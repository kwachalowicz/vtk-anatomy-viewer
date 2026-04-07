#!/usr/bin/env python3

import sys
from pathlib import Path
import vtk

ATLAS_DIR = Path(__file__).parent / "head-neck-2016-09"
MODELS_DIR = ATLAS_DIR / "models"
GRAYSCALE_DIR = ATLAS_DIR / "grayscale"
LABELS_DIR = ATLAS_DIR / "labels"
LUTS_DIR = ATLAS_DIR / "luts"


class NRRDReader:
    @staticmethod
    def read(filename: str):
        reader = vtk.vtkNrrdReader()
        reader.SetFileName(filename)
        reader.Update()
        image = reader.GetOutput()
        if image is None:
            print(f"Error: failed to read {filename}")
            return None
        image.SetSpacing(reader.GetOutput().GetSpacing())
        image.SetOrigin(reader.GetOutput().GetOrigin())
        return image


class AtlasViewer:
    def __init__(self):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetViewport(0.0, 0.0, 0.68, 1.0)
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1400, 900)
        self.render_window.SetWindowName("Head-Neck Atlas 3D Viewer")
        
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.Initialize()
        
        self.raw_data = None
        self.segmentation_data = None
        self.actors = {}
        self.outline_actor = None
        self.axes_actor = None
        self.slice_info = {}
        self.slider_widgets = []
        self.model_sliders = []
        self.scroll_offset = 0
        self.model_text_actors = []
        self.panel_background = None
        self.segmentation_overlays = {}
        self.model_colors = {}
        self.models_visible = True
        self.outline_only_mode = False
        self.show_segmentation = True
        self.contoured_actors = {}
        self.contoured_visible = False
        self.slice_view_renderers = {}
        self.slice_view_mappers = {}
        self.preview_seg_actors = {}
        
        self.setup_camera()
        
    def setup_camera(self):
        camera = self.renderer.GetActiveCamera()
        camera.SetViewUp(0, 0, 1)
        
    def reset_camera_to_data(self):
        if self.raw_data is None:
            return

        bounds = list(self.raw_data.GetBounds())
        self.renderer.ResetCamera(bounds)
        camera = self.renderer.GetActiveCamera()
        camera.Zoom(1.1)

        
    def load_atlas_data(self):
        grayscale_path = GRAYSCALE_DIR / "Osirix-Manix-255-res.nrrd"
        self.raw_data = NRRDReader.read(str(grayscale_path))
        
        labels_path = LABELS_DIR / "HN-Atlas-labels.nrrd"
        self.segmentation_data = NRRDReader.read(str(labels_path))
        
        self.load_color_table()
        
        if self.raw_data is not None:
            spacing = self.raw_data.GetSpacing()
            dims = self.raw_data.GetDimensions()
            centered_origin = (
                -0.5 * dims[0] * spacing[0],
                -0.5 * dims[1] * spacing[1],
                -0.5 * dims[2] * spacing[2],
            )
            self.raw_data.SetOrigin(centered_origin)
            if self.segmentation_data:
                self.segmentation_data.SetOrigin(centered_origin)

            self.add_outline()
            self.add_axes()
            self.reset_camera_to_data()
            return True
        
        return False
            
    def load_color_table(self):
        color_table_path = LUTS_DIR / "HeadAndNeckAtlas-training-colors.ctbl"
        
        if not color_table_path.exists():
            print("[!] Color table file not found")
            return
        
        with open(color_table_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        label_id = int(parts[0])
                        r = int(parts[2]) / 255.0
                        g = int(parts[3]) / 255.0
                        b = int(parts[4]) / 255.0
                        self.model_colors[label_id] = (r, g, b)
                    except (ValueError, IndexError):
                        continue
    
    def add_outline(self):
        if self.raw_data is None:
            return

        outline_filter = vtk.vtkOutlineFilter()
        outline_filter.SetInputData(self.raw_data)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(outline_filter.GetOutputPort())

        self.outline_actor = vtk.vtkActor()
        self.outline_actor.SetMapper(mapper)
        self.outline_actor.GetProperty().SetColor(0.5, 0.5, 0.5)
        self.outline_actor.GetProperty().SetLineWidth(2)

        self.renderer.AddActor(self.outline_actor)
    
    def add_axes(self):
        self.axes_actor = vtk.vtkAxesActor()
        self.axes_actor.AxisLabelsOff()
        self.axes_actor.SetTotalLength(20, 20, 20)
        self.axes_actor.GetXAxisCaptionActor2D().SetVisibility(False)
        self.axes_actor.GetYAxisCaptionActor2D().SetVisibility(False)
        self.axes_actor.GetZAxisCaptionActor2D().SetVisibility(False)
        self.renderer.AddActor(self.axes_actor)
    
    def add_orthogonal_planes(self):
        if self.raw_data is None:
            return
        
        extent = list(self.raw_data.GetExtent())
        x_mid = (extent[0] + extent[1]) // 2
        y_mid = (extent[2] + extent[3]) // 2
        z_mid = (extent[4] + extent[5]) // 2

        self.add_slice_actor("XY (Axial)", "axial", orientation="Z", slice_number=z_mid, extent=extent)
        self.add_slice_actor("YZ (Sagittal)", "sagittal", orientation="X", slice_number=x_mid, extent=extent)
        self.add_slice_actor("XZ (Coronal)", "coronal", orientation="Y", slice_number=y_mid, extent=extent)
        
        self.create_slice_previews(extent)
        self.add_preview_sliders(extent)

    def add_slice_actor(self, name, key: str, orientation: str, slice_number: int, extent):
        mapper = vtk.vtkImageSliceMapper()
        mapper.SetInputData(self.raw_data)
        if orientation == "Z":
            mapper.SetOrientationToZ()
        elif orientation == "X":
            mapper.SetOrientationToX()
        else:
            mapper.SetOrientationToY()
        mapper.SetSliceNumber(slice_number)

        slice_actor = vtk.vtkImageSlice()
        slice_actor.SetMapper(mapper)
        prop = slice_actor.GetProperty()
        prop.SetColorWindow(255)
        prop.SetColorLevel(127)
        prop.SetInterpolationTypeToNearest()

        self.renderer.AddActor(slice_actor)
        self.actors[f"plane_{name}"] = slice_actor
        self.slice_info[key] = {
            "mapper": mapper,
            "index": slice_number,
            "orientation": orientation,
            "extent": extent,
        }

    def on_slider_changed(self, key, widget):
        rep = widget.GetRepresentation()
        value = int(round(rep.GetValue()))
        self.set_slice_index(key, value)
    
    def add_opacity_sliders(self):
        model_names = sorted([k for k in self.actors.keys() if k.startswith('Model_')])
        if not model_names:
            return
        
        self.create_panel_background()
        
        self.all_models = model_names
        self.visible_slots = 15
        
        for slot_idx in range(self.visible_slots):
            self.create_opacity_slider_slot(slot_idx)
        
        self.add_scroll_slider()
        
        self.update_visible_sliders()
    
    def create_panel_background(self):
        points = vtk.vtkPoints()
        points.InsertNextPoint(0.01, 0.05, 0)
        points.InsertNextPoint(0.18, 0.05, 0)
        points.InsertNextPoint(0.18, 0.95, 0)
        points.InsertNextPoint(0.01, 0.95, 0)
        
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(4)
        for i in range(4):
            polygon.GetPointIds().SetId(i, i)
        
        polygons = vtk.vtkCellArray()
        polygons.InsertNextCell(polygon)
        
        polygonPolyData = vtk.vtkPolyData()
        polygonPolyData.SetPoints(points)
        polygonPolyData.SetPolys(polygons)
        
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(polygonPolyData)
        
        self.panel_background = vtk.vtkActor2D()
        self.panel_background.SetMapper(mapper)
        self.panel_background.GetProperty().SetColor(0.15, 0.15, 0.15)
        self.panel_background.GetProperty().SetOpacity(0.85)
        
        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToNormalizedViewport()
        mapper.SetTransformCoordinate(coordinate)
        
        self.renderer.AddViewProp(self.panel_background)
    
    def create_opacity_slider_slot(self, slot_idx):
        y_start = 0.88
        y_spacing = 0.055
        y_pos = y_start - slot_idx * y_spacing
        
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput("")
        text_actor.GetTextProperty().SetFontSize(10)
        text_actor.GetTextProperty().SetColor(0.9, 0.9, 0.9)
        text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        text_actor.GetPositionCoordinate().SetValue(0.015, y_pos + 0.01)
        self.renderer.AddViewProp(text_actor)
        self.model_text_actors.append(text_actor)
        
        rep = vtk.vtkSliderRepresentation2D()
        rep.SetMinimumValue(0.0)
        rep.SetMaximumValue(1.0)
        rep.SetValue(0.8)
        rep.GetSliderProperty().SetColor(0.3, 0.7, 0.9)
        rep.GetTubeProperty().SetColor(0.5, 0.5, 0.5)
        rep.GetCapProperty().SetColor(0.8, 0.8, 0.8)
        rep.SetSliderLength(0.012)
        rep.SetSliderWidth(0.015)
        rep.SetTubeWidth(0.002)
        rep.SetEndCapLength(0.006)
        rep.SetEndCapWidth(0.015)
        rep.ShowSliderLabelOff()
        rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep.GetPoint1Coordinate().SetValue(0.015, y_pos - 0.005)
        rep.GetPoint2Coordinate().SetValue(0.165, y_pos - 0.005)
        
        widget = vtk.vtkSliderWidget()
        widget.SetInteractor(self.interactor)
        widget.SetRepresentation(rep)
        widget.SetAnimationModeToJump()
        widget.EnabledOn()
        widget.AddObserver(
            "InteractionEvent",
            lambda obj, evt, idx=slot_idx: self.on_slot_opacity_changed(idx, obj),
        )
        self.model_sliders.append(widget)
    
    def add_scroll_slider(self):
        if len(self.all_models) <= self.visible_slots:
            return
        
        rep = vtk.vtkSliderRepresentation2D()
        rep.SetMinimumValue(0)
        rep.SetMaximumValue(len(self.all_models) - self.visible_slots)
        rep.SetValue(0)
        rep.GetSliderProperty().SetColor(0.6, 0.6, 0.6)
        rep.GetTubeProperty().SetColor(0.3, 0.3, 0.3)
        rep.GetCapProperty().SetColor(0.5, 0.5, 0.5)
        rep.SetSliderLength(0.02)
        rep.SetSliderWidth(0.015)
        rep.SetTubeWidth(0.003)
        rep.ShowSliderLabelOff()
        rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep.GetPoint1Coordinate().SetValue(0.005, 0.1)
        rep.GetPoint2Coordinate().SetValue(0.005, 0.9)
        
        scroll_widget = vtk.vtkSliderWidget()
        scroll_widget.SetInteractor(self.interactor)
        scroll_widget.SetRepresentation(rep)
        scroll_widget.SetAnimationModeToJump()
        scroll_widget.EnabledOn()
        scroll_widget.AddObserver(
            "InteractionEvent",
            lambda obj, evt: self.on_scroll_slider_changed(obj),
        )
        self.slider_widgets.append(scroll_widget)
    
    def update_visible_sliders(self):
        for slot_idx in range(self.visible_slots):
            model_idx = self.scroll_offset + slot_idx
            
            if model_idx < len(self.all_models):
                model_name = self.all_models[model_idx]
                actor = self.actors.get(model_name)
                
                display_name = model_name.replace('Model_', '').replace('_', ' ')[:20]
                self.model_text_actors[slot_idx].SetInput(display_name)
                self.model_text_actors[slot_idx].SetVisibility(True)
                
                if actor:
                    opacity = actor.GetProperty().GetOpacity()
                    self.model_sliders[slot_idx].GetRepresentation().SetValue(opacity)
                    self.model_sliders[slot_idx].SetEnabled(True)
                else:
                    self.model_sliders[slot_idx].SetEnabled(False)
            else:
                self.model_text_actors[slot_idx].SetVisibility(False)
                self.model_sliders[slot_idx].SetEnabled(False)
        
        self.render_window.Render()
    
    def on_slot_opacity_changed(self, slot_idx, widget):
        model_idx = self.scroll_offset + slot_idx
        if model_idx >= len(self.all_models):
            return
        
        model_name = self.all_models[model_idx]
        actor = self.actors.get(model_name)
        if not actor:
            return
        
        rep = widget.GetRepresentation()
        opacity = rep.GetValue()
        actor.GetProperty().SetOpacity(opacity)
        self.render_window.Render()
    
    def on_scroll_slider_changed(self, widget):
        rep = widget.GetRepresentation()
        self.scroll_offset = int(round(rep.GetValue()))
        self.update_visible_sliders()

    def create_slice_previews(self, extent):
        if self.raw_data is None:
            return

        viewports = {
            'axial': (0.70, 0.67, 0.98, 0.97),
            'sagittal': (0.70, 0.34, 0.98, 0.64),
            'coronal': (0.70, 0.01, 0.98, 0.31),
        }

        for key, vp in viewports.items():
            ren = vtk.vtkRenderer()
            ren.SetViewport(*vp)
            ren.SetBackground(0.05, 0.05, 0.05)
            ren.SetLayer(0)

            mapper = vtk.vtkImageSliceMapper()
            mapper.SetInputData(self.raw_data)
            if key == 'axial':
                mapper.SetOrientationToZ()
                mapper.SetSliceNumber((extent[4] + extent[5]) // 2)
            elif key == 'sagittal':
                mapper.SetOrientationToX()
                mapper.SetSliceNumber((extent[0] + extent[1]) // 2)
            else:
                mapper.SetOrientationToY()
                mapper.SetSliceNumber((extent[2] + extent[3]) // 2)

            actor = vtk.vtkImageSlice()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetInterpolationTypeToNearest()

            ren.AddActor(actor)

            cam = ren.GetActiveCamera()
            cam.ParallelProjectionOn()
            
            bounds = self.raw_data.GetBounds()
            cx = (bounds[0] + bounds[1]) / 2
            cy = (bounds[2] + bounds[3]) / 2
            cz = (bounds[4] + bounds[5]) / 2
            
            if key == 'axial':
                cam.SetPosition(cx, cy, bounds[5] + 100)
                cam.SetFocalPoint(cx, cy, cz)
                cam.SetViewUp(0, 1, 0)
            elif key == 'sagittal':
                cam.SetPosition(bounds[1] + 100, cy, cz)
                cam.SetFocalPoint(cx, cy, cz)
                cam.SetViewUp(0, 0, 1)
            else:
                cam.SetPosition(cx, bounds[3] + 100, cz)
                cam.SetFocalPoint(cx, cy, cz)
                cam.SetViewUp(0, 0, 1)
            
            size_x = bounds[1] - bounds[0]
            size_y = bounds[3] - bounds[2]
            size_z = bounds[5] - bounds[4]
            max_size = max(size_x, size_y, size_z)
            cam.SetParallelScale(max_size * 0.8)
            
            ren.ResetCameraClippingRange()

            self.render_window.AddRenderer(ren)
            self.slice_view_renderers[key] = ren
            self.slice_view_mappers[key] = mapper

    def add_preview_sliders(self, extent):
        slider_specs = [
            ("axial",    (extent[4], extent[5], (extent[4] + extent[5]) // 2), (0.72, 0.62), (0.98, 0.62)),
            ("sagittal", (extent[0], extent[1], (extent[0] + extent[1]) // 2), (0.72, 0.29), (0.98, 0.29)),
            ("coronal",  (extent[2], extent[3], (extent[2] + extent[3]) // 2), (0.72, 0.06), (0.98, 0.06)),
        ]

        for key, (vmin, vmax, vinit), p1, p2 in slider_specs:
            rep = vtk.vtkSliderRepresentation2D()
            rep.SetMinimumValue(vmin)
            rep.SetMaximumValue(vmax)
            rep.SetValue(vinit)
            rep.SetTitleText(key)
            rep.GetTitleProperty().SetColor(1, 1, 1)
            rep.GetLabelProperty().SetColor(1, 1, 1)
            rep.GetSliderProperty().SetColor(0.7, 0.5, 0.9)
            rep.GetTubeProperty().SetColor(0.6, 0.6, 0.6)
            rep.GetCapProperty().SetColor(0.9, 0.9, 0.9)
            rep.SetSliderLength(0.018)
            rep.SetSliderWidth(0.026)
            rep.SetTubeWidth(0.003)
            rep.SetEndCapLength(0.01)
            rep.SetEndCapWidth(0.028)
            rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
            rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
            rep.GetPoint1Coordinate().SetValue(*p1)
            rep.GetPoint2Coordinate().SetValue(*p2)

            widget = vtk.vtkSliderWidget()
            widget.SetInteractor(self.interactor)
            widget.SetRepresentation(rep)
            widget.SetAnimationModeToJump()
            widget.EnabledOn()
            widget.AddObserver(
                "InteractionEvent",
                lambda obj, evt, slice_key=key: self.on_slider_changed(slice_key, obj),
            )
            self.slider_widgets.append(widget)
    
    def add_preview_segmentation_overlays(self):
        if not self.segmentation_data or not self.model_colors:
            return
        
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(121)
        lut.SetRange(0, 120)
        
        for i in range(121):
            lut.SetTableValue(i, 0, 0, 0, 0)
        
        for label_id, rgb in self.model_colors.items():
            if label_id <= 120:
                lut.SetTableValue(label_id, rgb[0], rgb[1], rgb[2], 1.0)
        
        lut.SetTableValue(1, 0.8, 0.8, 0.8, 1.0)
        
        lut.Build()
        
        for key in ['axial', 'sagittal', 'coronal']:
            ren = self.slice_view_renderers.get(key)
            grayscale_mapper = self.slice_view_mappers.get(key)
            if not ren or not grayscale_mapper:
                continue
            
            slice_num = grayscale_mapper.GetSliceNumber()
            
            seg_mapper = vtk.vtkImageSliceMapper()
            seg_mapper.SetInputData(self.segmentation_data)
            
            if key == 'axial':
                seg_mapper.SetOrientationToZ()
            elif key == 'sagittal':
                seg_mapper.SetOrientationToX()
            else:
                seg_mapper.SetOrientationToY()
            
            seg_mapper.SetSliceNumber(slice_num)
            
            seg_actor = vtk.vtkImageSlice()
            seg_actor.SetMapper(seg_mapper)
            
            seg_actor.GetProperty().SetLookupTable(lut)
            seg_actor.GetProperty().UseLookupTableScalarRangeOn()
            seg_actor.GetProperty().SetInterpolationTypeToNearest()
            seg_actor.GetProperty().SetOpacity(0.6)
            
            ren.AddActor(seg_actor)
            self.preview_seg_actors[key] = {'actor': seg_actor, 'mapper': seg_mapper}
        
        self.render_window.Render()
    
    def add_segmentation_overlays(self):
        if not self.segmentation_data or not self.model_colors:
            return
        
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(121)
        lut.SetRange(0, 120)
        
        for i in range(121):
            lut.SetTableValue(i, 0, 0, 0, 0)
        
        mapped_count = 0
        for label_id, rgb in self.model_colors.items():
            if label_id <= 120:
                lut.SetTableValue(label_id, rgb[0], rgb[1], rgb[2], 1.0)
                mapped_count += 1
        
        lut.SetTableValue(1, 0.8, 0.8, 0.8, 1.0)
        
        lut.Build()
        
        for key in ['axial', 'sagittal', 'coronal']:
            info = self.slice_info.get(key)
            if not info:
                continue
            
            seg_mapper = vtk.vtkImageSliceMapper()
            seg_mapper.SetInputData(self.segmentation_data)
            
            if info['orientation'] == 'Z':
                seg_mapper.SetOrientationToZ()
            elif info['orientation'] == 'X':
                seg_mapper.SetOrientationToX()
            else:
                seg_mapper.SetOrientationToY()
            
            seg_mapper.SetSliceNumber(info['index'])
            
            seg_actor = vtk.vtkImageSlice()
            seg_actor.SetMapper(seg_mapper)
            
            seg_actor.GetProperty().SetLookupTable(lut)
            seg_actor.GetProperty().UseLookupTableScalarRangeOn()
            seg_actor.GetProperty().SetInterpolationTypeToNearest()
            seg_actor.GetProperty().SetOpacity(1.0)
            
            self.renderer.AddActor(seg_actor)
            self.segmentation_overlays[key] = {
                'actor': seg_actor,
                'mapper': seg_mapper
            }
    
    def load_vtk_model(self, model_path, name, color=None):
        filepath = MODELS_DIR / model_path
        
        if not filepath.exists():
            return None
        
        try:
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(str(filepath))
            reader.Update()
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            if color:
                actor.GetProperty().SetColor(*color)
            else:
                actor.GetProperty().SetColor(0.8, 0.8, 0.8)

            actor.GetProperty().SetOpacity(0.8)
            
            self.renderer.AddActor(actor)
            self.actors[name] = actor
            
            return actor
        except Exception as e:
            return None
    
    def add_key_structures(self):
        vtk_files = sorted([p for p in MODELS_DIR.glob('Model_*.vtk')])
        
        for path in vtk_files:
            name = path.stem
            try:
                label_id = int(name.split('_')[1])
                color = self.model_colors.get(label_id)
                
                if not color:
                    color = (0.7, 0.7, 0.7)
                    self.model_colors[label_id] = color
            except (IndexError, ValueError):
                color = (0.7, 0.7, 0.7)
            
            self.load_vtk_model(path.name, name, color)
        
        self.add_opacity_sliders()
        
        self.add_segmentation_overlays()
        
        self.add_preview_segmentation_overlays()
    
    def setup_interactor_style(self):
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        self.interactor.AddObserver('KeyPressEvent', self.on_key_press)
    
    def on_key_press(self, obj, event):
        key = obj.GetKeySym()
        
        if key == "h":
            self.print_help()
        elif key == "r":
            self.renderer.ResetCamera()
            self.render_window.Render()
        elif key == "w":
            for actor in self.actors.values():
                actor.GetProperty().SetRepresentationToWireframe()
            self.render_window.Render()
        elif key == "s":
            for actor in self.actors.values():
                actor.GetProperty().SetRepresentationToSurface()
            self.render_window.Render()
        elif key == "t":
            self.models_visible = not self.models_visible
            for name, actor in self.actors.items():
                if name.startswith('Model_'):
                    actor.SetVisibility(self.models_visible)
            self.render_window.Render()
        elif key == "o":
            self.outline_only_mode = not self.outline_only_mode
            
            for name, actor in self.actors.items():
                if name.startswith('Model_'):
                    actor.SetVisibility(not self.outline_only_mode)
                elif name.startswith('plane_'):
                    actor.SetVisibility(not self.outline_only_mode)
            
            for overlay in self.segmentation_overlays.values():
                overlay['actor'].SetVisibility(not self.outline_only_mode)
            
            self.render_window.Render()
        elif key == "g":
            self.show_segmentation = not self.show_segmentation
            
            for overlay in self.segmentation_overlays.values():
                overlay['actor'].SetVisibility(self.show_segmentation)
            
            for overlay in self.preview_seg_actors.values():
                overlay['actor'].SetVisibility(self.show_segmentation)
            
            self.render_window.Render()
        elif key == "c":
            self.toggle_contoured_models()
        elif key == "Up":
            self.shift_slice("axial", +1)
        elif key == "Down":
            self.shift_slice("axial", -1)
        elif key == "Right":
            self.shift_slice("coronal", +1)
        elif key == "Left":
            self.shift_slice("coronal", -1)
        elif key == "Prior":
            self.shift_slice("sagittal", +1)
        elif key == "Next":
            self.shift_slice("sagittal", -1)
        elif key == "q" or key == "Escape":
            self.interactor.ExitCallback()

    def shift_slice(self, key: str, delta: int):
        info = self.slice_info.get(key)
        if not info:
            return
        self.set_slice_index(key, info["index"] + delta)

    def set_slice_index(self, key: str, idx: int):
        info = self.slice_info.get(key)
        if not info:
            return
        ext = info["extent"]
        if info["orientation"] == "Z":
            lo, hi = ext[4], ext[5]
        elif info["orientation"] == "Y":
            lo, hi = ext[2], ext[3]
        else:
            lo, hi = ext[0], ext[1]
        idx = max(lo, min(hi, idx))
        if idx == info["index"]:
            return
        info["index"] = idx
        info["mapper"].SetSliceNumber(idx)
        info["mapper"].Update()
        
        if key in self.segmentation_overlays:
            self.segmentation_overlays[key]['mapper'].SetSliceNumber(idx)
            self.segmentation_overlays[key]['mapper'].Update()

        if key in self.slice_view_mappers:
            self.slice_view_mappers[key].SetSliceNumber(idx)
            self.slice_view_mappers[key].Update()
            
            if key in self.preview_seg_actors:
                self.preview_seg_actors[key]['mapper'].SetSliceNumber(idx)
                self.preview_seg_actors[key]['mapper'].Update()
        
        for slider in self.slider_widgets:
            if slider.GetRepresentation().GetTitleText() == key:
                slider.GetRepresentation().SetValue(idx)
        self.render_window.Render()
    
    def create_contoured_surface(self, label_id, smoothing_iterations=15, decimation_target=0.5):
        if self.segmentation_data is None:
            return None
        
        marching_cubes = vtk.vtkDiscreteMarchingCubes()
        marching_cubes.SetInputData(self.segmentation_data)
        marching_cubes.SetValue(0, label_id)
        marching_cubes.Update()
        
        num_points = marching_cubes.GetOutput().GetNumberOfPoints()
        
        if num_points == 0:
            return None
        
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(marching_cubes.GetOutputPort())
        smoother.SetNumberOfIterations(smoothing_iterations)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetFeatureAngle(120.0)
        smoother.SetPassBand(0.001)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        
        decimator = vtk.vtkDecimatePro()
        decimator.SetInputConnection(smoother.GetOutputPort())
        decimator.SetTargetReduction(decimation_target)
        decimator.PreserveTopologyOn()
        decimator.Update()
        
        return decimator.GetOutput()
    
    def toggle_contoured_models(self):
        if not self.contoured_visible:
            if not self.contoured_actors:
                print("\n=== Generowanie wygładzonych modeli ===")
                all_labels = sorted(self.model_colors.keys())
                
                for label_id in all_labels:
                    
                    poly_data = self.create_contoured_surface(
                        label_id,
                        smoothing_iterations=20,
                        decimation_target=0.6
                    )
                    
                    if poly_data is None:
                        continue
                    
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputData(poly_data)
                    
                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    
                    color = self.model_colors[label_id]
                    actor.GetProperty().SetColor(*color)
                    actor.GetProperty().SetOpacity(0.9)
                    
                    actor.GetProperty().SetSpecular(0.3)
                    actor.GetProperty().SetSpecularPower(20)
                    
                    actor.SetVisibility(False)
                    self.renderer.AddActor(actor)
                    self.contoured_actors[label_id] = actor
            
            for actor in self.contoured_actors.values():
                actor.SetVisibility(True)
            for name, actor in self.actors.items():
                if name.startswith('Model_'):
                    actor.SetVisibility(False)
            
            self.contoured_visible = True
        else:
            for actor in self.contoured_actors.values():
                actor.SetVisibility(False)
            for name, actor in self.actors.items():
                if name.startswith('Model_'):
                    actor.SetVisibility(self.models_visible)
            
            self.contoured_visible = False
        
        self.render_window.Render()
    
    def print_help(self):
        print("""
╔════════════════════════════════════════════╗
║    Wizualizacja Atlasu Głowy i Szyi        ║
╠════════════════════════════════════════════╣
║  Sterowanie myszką:                        ║
║    Lewy + przeciągnij  - Obracanie widoku  ║
║    Prawy + przeciągnij - Powiększanie      ║
║    Środkowy + przeciąg - Przesuwanie       ║
║                                            ║
║  Skróty klawiszowe:                        ║
║    h - Pokaż tę pomoc                      ║
║    r - Zresetuj kamerę                     ║
║    w - Tryb wireframe                      ║
║    s - Tryb powierzchniowy (surface)       ║
║    o - Tylko kontur (outline)              ║
║    g - Przełącz segmentację/skala szarości ║
║    c - Przełącz modele wygładzone          ║
║    t - Przełącz modele 3D wł/wył           ║
║    ↑/↓ - Przesuń przekrój osiowy           ║
║    ←/→ - Przesuń przekrój koronalny        ║
║    PgUp/PgDn - Przesuń przekrój strzałkowy ║
║    q lub Esc - Wyjście                     ║
║  Suwaki:                                   ║
║    Pod podglądami (prawa) - wybór przekroju║
║    Lewy panel - przezroczystość modeli     ║
║  Mysz:                                     ║
║    Kółko myszy - powiększanie/pomniejszanie║
╚════════════════════════════════════════════╝
        """)
    
    def run(self):
        if not self.load_atlas_data():
            return False
        
        self.add_orthogonal_planes()
        self.add_key_structures()

        self.renderer.ResetCamera()
        
        self.setup_interactor_style()
        
        self.print_help()
        
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        self.render_window.Render()
        self.interactor.Start()
        
        return True


def main():
    if not ATLAS_DIR.exists():
        sys.exit(1)
    
    viewer = AtlasViewer()
    if not viewer.run():
        sys.exit(1)


if __name__ == "__main__":
    main()
