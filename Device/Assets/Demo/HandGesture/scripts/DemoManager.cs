using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;

public class DemoManager : MonoBehaviour
{
    public GameObject TargetApp;
    public GameObject Grid;
    Dictionary<string, GameObject> gridDict = new Dictionary<string, GameObject>();
    GameObject curButton;
    int curIndex = 5;
    string InputString = "";
    public string mode = "select";
    public Slider slider;

    public GameObject Canvas;
    Dictionary<string, GameObject> musicDict = new Dictionary<string, GameObject>();
    public List<Texture> TextureList = new List<Texture>();
	private RawImage img;

    int textureIdx = 0;

    public GameObject speaker;

    private float _currentScale = 1000.0f;
    private float InitScale = 1000.0f;
    private const int FramesCount = 100;
    private const float AnimationTimeSeconds = 1.0f;
    private float varAnimTime = 0.8f;
 
    private bool _upScale = true;

    private bool scaling = false;
    private float maxScale = 50.0f;

    bool flag_activate = false;

    // Start is called before the first frame update
    void Start()
    {
        TargetApp.SetActive(flag_activate);

        //get child by name under Grid
        foreach (Transform child in Grid.transform)
        {
            gridDict[child.name] = child.gameObject;
        }
        curButton = gridDict[curIndex.ToString()];
        Animator animator = curButton.GetComponent<Animator>();
        animator.SetBool("Highlighted", true);

        //get child by name under Grid
        foreach (Transform child in Canvas.transform)
        {
            musicDict[child.name] = child.gameObject;
        }
        img = musicDict["album"].GetComponent<RawImage>();
        
        img.texture = (Texture)TextureList[textureIdx];
        
    }

    // Update is called once per frame
    void Update()
    {
        if (mode == "select")
        {
            switch (InputString)
            {
                case "up":
                    NormalButton(curButton);
                    InputString = "";
                    curIndex -= 3;
                    if (curIndex <= 0)
                        curIndex += 9;
                    curButton = gridDict[curIndex.ToString()];
                    HighlightButton(curButton);
                    break;
                case "down":
                    NormalButton(curButton);
                    InputString = "";
                    curIndex += 3;
                    if (curIndex > 9)
                        curIndex -= 9;
                    curButton = gridDict[curIndex.ToString()];
                    HighlightButton(curButton);
                    break;
                case "left":
                    NormalButton(curButton);
                    InputString = "";
                    curIndex -= 1;
                    if (curIndex == 0)
                        curIndex += 9;
                    curButton = gridDict[curIndex.ToString()];
                    HighlightButton(curButton);
                    break;
                case "right":
                    NormalButton(curButton);
                    InputString = "";
                    curIndex += 1;
                    if (curIndex > 9)
                        curIndex -= 9;
                    curButton = gridDict[curIndex.ToString()];
                    HighlightButton(curButton);
                    break;
                case "tap":                   
                    NormalButton(curButton);
                    InputString = "";
                    PressButton(curButton);
                    break;
                case "cclock":
                    TargetApp.SetActive(false);
                    InputString = "";
                    break;
                case "clock":
                    TargetApp.SetActive(true);
                    InputString = "";
                    break;
                default:
                    break;
            }
        }

        if (mode == "music")
        { 
            switch (InputString)
            {                
                case "up":
                    InputString = "";
                    break;
                case "down":
                    InputString = "";
                    break;
                case "left":
                    InputString = "";
                    textureIdx -= 1;
                    if (textureIdx < 0)
                        textureIdx = 2;                                           
                    img = (RawImage)musicDict["album"].GetComponent<RawImage>();
                    img.texture = (Texture)TextureList[textureIdx];
                    break;
                case "right":
                    InputString = "";
                    textureIdx += 1;
                    if (textureIdx > 2)
                        textureIdx = 0;
                             
                    img = (RawImage)musicDict["album"].GetComponent<RawImage>();
                    img.texture = (Texture)TextureList[textureIdx];
                    break;
                case "tap":
                    flag_activate = !flag_activate;
                    TargetApp.SetActive(flag_activate);
                    InputString = "";
                    break;
                case "cclock":
                    StartCoroutine(DecreaseVolume());
                    InputString = "";
                    break;
                case "clock":
                    StartCoroutine(IncreaseVolume());
                    InputString = "";
                    break;
                default:
                    InputString = "";
                    break;
            }
        }

        if (!scaling)
        {            
            if (_upScale)
            {
                Debug.Log("upscale : " + slider.value);
                StartCoroutine(ScaleUpSpeaker());            
            }
            else
            {
                Debug.Log("downscale : " + slider.value);
                StartCoroutine(ScaleDownSpeaker());            
            }
        }
        
    }


    private IEnumerator ScaleUpSpeaker()
    {
        // float scale = ;

        
        float TargetScale = InitScale + maxScale*slider.value + 5.0f;
        
        float animTime = AnimationTimeSeconds - varAnimTime * slider.value;

        float _deltaTime = animTime/FramesCount;

        float _dx = (TargetScale - _currentScale)/FramesCount;

        while (true)
        {
            _currentScale += _dx;            
            speaker.transform.localScale = Vector3.one * _currentScale;        

            if (_currentScale > TargetScale)
            {
                _upScale = false;
                scaling = false;
                break;
            }
            yield return new WaitForSeconds(_deltaTime);    
        }
        
    }
    
    private IEnumerator ScaleDownSpeaker()
    {
        // float scale = slider.value;
        float TargetScale = InitScale - maxScale*slider.value - 5.0f;

        float animTime = AnimationTimeSeconds - varAnimTime * slider.value;
        float _deltaTime = animTime/FramesCount;
        float _dx = (TargetScale - _currentScale)/FramesCount;

        while (true)
        {
            _currentScale += _dx;            
            speaker.transform.localScale = Vector3.one * _currentScale;        

            if (_currentScale < TargetScale)
            {
                _upScale = true;
                scaling = false;               
                break;
            }
            yield return new WaitForSeconds(_deltaTime);    
        }             
    }
    
    
    IEnumerator IncreaseVolume()
    {
        for (int i =0; i < 40; i++)
        {
            if (slider.value < 1)
            {
                slider.value += 0.005f;
                yield return new WaitForSeconds(0.01f);
            }
        }    
    }

    IEnumerator DecreaseVolume()
    {
        for (int i = 0; i < 40; i++)
        {
            if (slider.value > 0)
            {
                slider.value -= 0.005f;
                yield return new WaitForSeconds(0.01f);
            }
        }
    }


    void NormalButton(GameObject curButton)
    {
        Animator animator = curButton.GetComponent<Animator>();
        animator.SetBool("Highlighted", false);
        animator.SetBool("Pressed", false);
        animator.SetBool("Normal", true);
    }

    void HighlightButton(GameObject  curButton)
    {
        Animator animator = curButton.GetComponent<Animator>();
        animator.SetBool("Highlighted", true);
    }

    void PressButton(GameObject curButton)
    {
        Animator animator = curButton.GetComponent<Animator>();
        animator.SetBool("Pressed", true);
    }

    public void GetInputMessage(string inputMessage)
    {
        InputString = inputMessage;
    }
}
