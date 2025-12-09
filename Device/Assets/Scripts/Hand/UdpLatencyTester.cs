using System;
using System.Net;
using System.Net.Sockets;
using UnityEngine;
using TMPro;
using System.Collections.Generic;

public class UdpLatencyTester : MonoBehaviour
{
    public string serverIP = "192.168.50.147"; // 파이썬 실행 장치 IP
    public int port = 5005;
    public TextMeshProUGUI latencyText;

    private UdpClient udpClient;
    private IPEndPoint remoteEndPoint;

    // 🔹 최근 latency 저장용
    private Queue<double> latencyQueue = new Queue<double>();
    private const int maxSamples = 10;

    void Start()
    {
        udpClient = new UdpClient();
        remoteEndPoint = new IPEndPoint(IPAddress.Parse(serverIP), port);

        // 0.5초마다 latency 측정
        InvokeRepeating(nameof(SendPing), 1f, 0.5f);
    }

    void SendPing()
    {
        long sendTime = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        byte[] data = BitConverter.GetBytes(sendTime);

        udpClient.Send(data, data.Length, remoteEndPoint);
        udpClient.BeginReceive(OnReceive, sendTime);
    }

    void OnReceive(IAsyncResult ar)
    {
        try
        {
            IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
            byte[] data = udpClient.EndReceive(ar, ref anyIP);

            long sendTime = (long)ar.AsyncState;
            long receiveTime = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

            long rtt = receiveTime - sendTime; // 왕복 시간
            double latency = rtt / 2.0;        // 편도 지연 추정

            // 🔹 최근 10개만 유지
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
                double sum = 0;
                foreach (var l in latencyQueue)
                    sum += l;
                avgLatency = sum / latencyQueue.Count;
            }

            UnityMainThreadDispatcher.Instance().Enqueue(() =>
            {
                latencyText.text = $"Latency(avg {latencyQueue.Count}): {avgLatency:F2} ms";
            });
        }
        catch (Exception ex)
        {
            Debug.Log("Receive error: " + ex.Message);
        }
    }

    void OnApplicationQuit()
    {
        udpClient?.Close();
    }
}