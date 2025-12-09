using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class HandModuleTemplate : MonoBehaviour
{
    float[] handPose3D = new float[63];
    int gestureIdx = -1;    
    string inputGesture = "";
    List<string> Gestures = new List<string>(new string[]{"up","down","left","right","clock","cclock","tap"});

    GameObject hand, handMesh;
    GameObject[] handJoints = new GameObject[21];
    Vector3[] joint3DWorldArray = new Vector3[21];
    
    Matrix4x4 camIntrinsic = Matrix4x4.identity;
    Matrix4x4 camExtrinsic = Matrix4x4.identity;
    Matrix4x4 inv = Matrix4x4.identity;
    float rootDepth = 0.0f;
    int imgHeight = 0;
    int imgWidth = 0;
    float defaultDepth = 0.370f;

    // Start is called before the first frame update
    void Start()
    {
        // load handMesh prefabs
        // handMesh = Resources.Load("Hand/HandSkeleton") as GameObject;
        // hand = Instantiate(handMesh, new Vector3(0, 0, 0), Quaternion.identity) as GameObject;
        // for (int i = 0; i < 21; i++){
        //     handJoints[i] = hand.transform.GetChild(i).gameObject;
        // }

        // // TODO : initialize cam variables to visualize hand
        // Matrix4x4 camIntrinsic = Matrix4x4.identity;
        // Matrix4x4 camExtrinsic = Matrix4x4.identity;
        // inv = camIntrinsic * camExtrinsic; 
        // inv = inv.inverse;

        imgWidth = 640;
        imgHeight = 480;        

        rootDepth = defaultDepth;   //  TODO. update from value
    }

    // Update is called once per frame
    void Update()
    {
        // TODO : visualize hand 3D mesh
        // Compute hand joint on world coordinate
        // for (int i = 0; i < 21; i++)
        // {
        //     // pixel 단위. u starts from left, v start from bottom.
        //     // Vector3 joint3DImage = new Vector3(handData.joints[i].u, imgHeight - handData.joints[i].v, 1);
        //     Vector3 joint3DImage = new Vector3(handPose3D[i*3], imgHeight - handPose3D[i*3+1], 1);
            
        //     joint3DImage *= rootDepth + handPose3D[i*3+2];

        //     Vector3 joint3DWorld = inv.MultiplyPoint3x4(joint3DImage);
        //     joint3DWorldArray[i] = new Vector3(joint3DWorld.x, joint3DWorld.y, joint3DWorld.z);   
        // }
        // // apply to hand mesh
        // for (int i = 0; i < 21; i++)
        // {
        //     handJoints[i].transform.position = joint3DWorldArray[i];            
        // }


        // // implement functions for given gesture input
        switch (inputGesture)
        {
            case "up":
                inputGesture = "";
                break;
            case "down":
                inputGesture = "";
                break;
            case "left":
                inputGesture = "";
                break;
            case "right":
                inputGesture = "";
                break;
            case "clock":                   
                inputGesture = "";
                break;
            case "cclock":
                inputGesture = "";
                break;
            case "tap":
                inputGesture = "";
                break;
            default:
                break;
        }
    }

    public void GetInputMessage(double[] msgPose, int msgGesture)
    {
        float[] handPose3D = msgPose.Select(x => (float)x).ToArray();
        gestureIdx = msgGesture;
        inputGesture = Gestures[gestureIdx];
    }
}
