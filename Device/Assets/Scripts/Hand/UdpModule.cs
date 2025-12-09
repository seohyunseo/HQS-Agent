using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using Microsoft.MixedReality.Toolkit.Experimental.UI;
using System;
using System.Collections.Generic;
using System.Linq;
using TMPro;


public class UdpModule : MonoBehaviour
{
    private UdpClient udpClient;
    private Thread receiveThread;
    private int port = 5005;
    public HandModuleTemplate handTemplate;
    public TextMeshProUGUI timeText;
    string debugText;

    private Queue<double> latencyQueue = new Queue<double>();
    private const int maxSamples = 10;


    void Start()
    {
        UnityMainThreadDispatcher.Instance().Enqueue(() =>
        {
            timeText.text = "Dispatcher Test OK";
        });

        udpClient = new UdpClient(port);
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    private void ReceiveData()
    {
        while (true)
        {
            try
            {
                IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, port);
                byte[] data = udpClient.Receive(ref remoteEndPoint);

                double[] dataArray = new double[data.Length / 8];
                Buffer.BlockCopy(data, 0, dataArray, 0, data.Length);

                Debug.Log("Received data: " + string.Join(", ", dataArray));
                debugText = string.Join(", ", dataArray);

                // python : send_data = outs.flatten().tolist() + [float(valid_gesture_idx), float(time.time()*1000)]
                double[] handPose = new double[63];
                Array.Copy(dataArray, 0, handPose, 0, 63);

                double gestureIdx = dataArray[63];
                double pythonTime = dataArray[64];


                double unityTime = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(); // ms 단위
                double latency = unityTime - pythonTime;

                lock (latencyQueue)
                {
                    latencyQueue.Enqueue(latency);
                    if (latencyQueue.Count > maxSamples)
                        latencyQueue.Dequeue();
                }

                // 🔹 평균 계산
                double avgLatency;
                lock (latencyQueue)
                {
                    avgLatency = 0;
                    foreach (var l in latencyQueue)
                        avgLatency += l;
                    avgLatency /= latencyQueue.Count;
                }


                // UI 업데이트는 메인 스레드에서만 가능하므로 변수에 저장
                UnityMainThreadDispatcher.Instance().Enqueue(() =>
                {
                    timeText.text = $"Latency : {avgLatency:F2} ms";

                });


            }
            catch (SocketException ex)
            {
                Debug.Log("SocketException: " + ex.Message);
            }
        }
    }

    void OnApplicationQuit()
    {
        if (receiveThread != null)
        {
            receiveThread.Abort();
        }
        udpClient.Close();
    }
}