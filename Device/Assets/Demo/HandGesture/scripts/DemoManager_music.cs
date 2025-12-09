using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class DemoManager_music : MonoBehaviour
{
    public GameObject speaker;
    Dictionary<string, GameObject> gridDict = new Dictionary<string, GameObject>();
    GameObject curButton;
    int curIndex = 5;
    string InputString = "";
    // Start is called before the first frame update
    void Start()
    {
        Animator animator = speaker.GetComponent<Animator>();
        animator.SetBool("Highlighted", true);
    }

    // Update is called once per frame
    void Update()
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
            case "tab":
                NormalButton(curButton);
                InputString = "";
                PressButton(curButton);
                break;
            default:
                break;
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
