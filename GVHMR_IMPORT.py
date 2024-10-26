import bpy
import os
import pickle
import numpy as np
from mathutils import Matrix, Vector

bl_info = {
    "name": "SMPL_FBX Loader",
    "blender": (2, 90, 0),
    "category": "Object",
}

class SMPLLoaderPanel(bpy.types.Panel):
    bl_label = "SMPL_FBX Loader"
    bl_idname = "VIEW3D_PT_smpl_loader"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'GVHMR IMPORT'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # File browser for the smpl_model and PKL FILE
        layout.prop(scene, "smpl_model_path")
        layout.prop(scene, "result_file_path")
        
        # Button to execute the operator
        layout.operator("object.process_smpl_data")

class ProcessSMPLDataOperator(bpy.types.Operator):
    bl_idname = "object.process_smpl_data"
    bl_label = "Process SMPL Data"

    def execute(self, context):
        scene = context.scene
        smpl_model = scene.smpl_model_path
        result_file = scene.result_file_path
        
        # Ensure both files are selected
        if not smpl_model or not result_file:
            self.report({'ERROR'}, "Please select both the SMPL_FBX and the PKL FILE.")
            return {'CANCELLED'}

        # Execute the new processing function with the selected files
        self.process_smpl_data(smpl_model, result_file)
        
        return {'FINISHED'}

    def process_smpl_data(self, smpl_model, result_file):
        # New functionality for processing SMPL data
        with open(result_file, 'rb') as handle:
            results = pickle.load(handle)

        part_match_custom_less2 = {
            'root': 'root', 'bone_00': 'Pelvis', 'bone_01': 'L_Hip', 'bone_02': 'R_Hip', 
            'bone_03': 'Spine1', 'bone_04': 'L_Knee', 'bone_05': 'R_Knee', 'bone_06': 'Spine2', 
            'bone_07': 'L_Ankle', 'bone_08': 'R_Ankle', 'bone_09': 'Spine3', 'bone_10': 'L_Foot', 
            'bone_11': 'R_Foot', 'bone_12': 'Neck', 'bone_13': 'L_Collar', 'bone_14': 'R_Collar', 
            'bone_15': 'Head', 'bone_16': 'L_Shoulder', 'bone_17': 'R_Shoulder', 'bone_18': 'L_Elbow', 
            'bone_19': 'R_Elbow', 'bone_20': 'L_Wrist', 'bone_21': 'R_Wrist', 'bone_22': 'L_Hand', 'bone_23': 'R_Hand'
        }

        def Rodrigues(rotvec):
            theta = np.linalg.norm(rotvec)
            r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
            cost = np.cos(theta)
            mat = np.asarray([[0, -r[2], r[1]],
                              [r[2], 0, -r[0]],
                              [-r[1], r[0], 0]], dtype=object)
            return cost*np.eye(3) + (1 - cost)*r.dot(r.T) + np.sin(theta)*mat

        def rodrigues2bshapes(pose):
            rod_rots = np.asarray(pose).reshape(22, 3)
            mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]
            bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel()
                                      for mat_rot in mat_rots[1:]])
            return mat_rots, bshapes

        def get_global_pose(global_pose, arm_ob, frame=None):
            arm_ob.pose.bones['m_avg_root'].rotation_quaternion.w = 0.0
            arm_ob.pose.bones['m_avg_root'].rotation_quaternion.x = -1.0
            bone = arm_ob.pose.bones['m_avg_Pelvis']
            root_orig = arm_ob.pose.bones['m_avg_root'].rotation_quaternion
            mw_orig = arm_ob.matrix_world.to_quaternion()
            pelvis_quat = Matrix(global_pose[0]).to_quaternion()
            bone.rotation_quaternion = pelvis_quat
            bone.keyframe_insert('rotation_quaternion', frame=frame)
            pelvis_applyied = arm_ob.pose.bones['m_avg_Pelvis'].rotation_quaternion
            bpy.context.view_layer.update()
            rot_world_orig = root_orig @ pelvis_applyied @ mw_orig
            return rot_world_orig

        def apply_trans_pose_shape(trans, body_pose, arm_ob, obname, frame=None):
            mrots, bsh = rodrigues2bshapes(body_pose)
            part_bones = part_match_custom_less2
            trans = Vector((trans[0], trans[1] - 1.5, trans[2]))
            arm_ob.pose.bones['m_avg_Pelvis'].location = trans
            arm_ob.pose.bones['m_avg_Pelvis'].keyframe_insert('location', frame=frame)
            arm_ob.pose.bones['m_avg_root'].rotation_quaternion.w = 0.0
            arm_ob.pose.bones['m_avg_root'].rotation_quaternion.x = -1.0

            for ibone, mrot in enumerate(mrots):
                if ibone < 22:
                    bone = arm_ob.pose.bones['m_avg_' + part_bones['bone_%02d' % ibone]]
                    bone.rotation_quaternion = Matrix(mrot).to_quaternion()
                    if frame is not None:
                        bone.keyframe_insert('rotation_quaternion', frame=frame)

        def init_scene(scene, params, gender='male', angle=0):
            path_fbx = smpl_model
            bpy.ops.import_scene.fbx(filepath=path_fbx, axis_forward='-Y', axis_up='-Z', global_scale=100)
            arm_ob = bpy.context.selected_objects[0]
            obj_gender = 'm'
            obname = '%s_avg' % obj_gender
            ob = bpy.data.objects[obname]
            bpy.ops.object.select_all(action='DESELECT')
            cam_ob = ''
            arm_ob.animation_data_clear()
            return ob, obname, arm_ob, cam_ob

        print("Processing SMPL Data from:", smpl_model)
        print("Using results from:", result_file)

        params = []
        object_name = 'm_avg'
        obj_gender = 'm'
        scene = bpy.data.scenes['Scene']
        ob, obname, arm_ob, cam_ob = init_scene(scene, params, obj_gender)

        qtd_frames = len(results['smpl_params_global']['transl'])
        for fframe in range(0, qtd_frames):
            bpy.context.scene.frame_set(fframe)
            trans = results['smpl_params_global']['transl'][fframe]
            global_orient = results['smpl_params_global']['global_orient'][fframe]
            body_pose = results['smpl_params_global']['body_pose'][fframe]
            body_pose_fim = body_pose.reshape(int(len(body_pose)/3), 3)
            final_body_pose = np.vstack([global_orient, body_pose_fim])
            apply_trans_pose_shape(Vector(trans), final_body_pose, arm_ob, obname, fframe)
            bpy.context.view_layer.update()

        arm_ob.pose.bones['m_avg_root'].rotation_quaternion.w = 1.0
        arm_ob.pose.bones['m_avg_root'].rotation_quaternion.x = 0.0
        arm_ob.pose.bones['m_avg_root'].rotation_quaternion.y = 0.0
        arm_ob.pose.bones['m_avg_root'].rotation_quaternion.z = 0.0

        arm_ob.pose.bones['m_avg_Pelvis'].constraints.new('COPY_LOCATION')
        arm_ob.pose.bones["m_avg_Pelvis"].constraints[0].target = arm_ob
        arm_ob.pose.bones["m_avg_Pelvis"].constraints[0].subtarget = "m_avg_Pelvis"

        arm_ob.pose.bones['m_avg_Pelvis'].constraints.new('COPY_ROTATION')
        arm_ob.pose.bones["m_avg_Pelvis"].constraints[1].target = arm_ob
        arm_ob.pose.bones["m_avg_Pelvis"].constraints[1].subtarget = "m_avg_Pelvis"

def register():
    bpy.utils.register_class(SMPLLoaderPanel)
    bpy.utils.register_class(ProcessSMPLDataOperator)
    
    bpy.types.Scene.smpl_model_path = bpy.props.StringProperty(name="SMPL_FBX", subtype='FILE_PATH')
    bpy.types.Scene.result_file_path = bpy.props.StringProperty(name="PKL FILE", subtype='FILE_PATH')

def unregister():
    bpy.utils.unregister_class(SMPLLoaderPanel)
    bpy.utils.unregister_class(ProcessSMPLData)