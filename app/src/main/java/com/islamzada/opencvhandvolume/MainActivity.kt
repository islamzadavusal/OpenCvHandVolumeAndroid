package com.islamzada.opencvhandvolume

import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.media.AudioManager
import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import android.widget.SeekBar
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.islamzada.opencvhandvolume.databinding.ActivityMainBinding
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfInt4
import org.opencv.core.MatOfPoint
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import java.io.File
import java.io.FileOutputStream
import kotlin.math.acos
import kotlin.math.pow
import kotlin.math.sqrt

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    private lateinit var binding: ActivityMainBinding
    private lateinit var audioManager: AudioManager
    private var cascadeClassifier: CascadeClassifier? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (OpenCVLoader.initLocal()) {
            Log.i("OpenCV", "OpenCV successfully loaded")
            loadCascadeClassifier()
            initializeCamera()
        } else {
            Log.e("OpenCV", "Failed to load OpenCV")
        }

        audioManager = getSystemService(AUDIO_SERVICE) as AudioManager
        setupVolumeControl()
        checkCameraPermission()
    }

    private fun checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.CAMERA), 1)
        } else {
            initializeCamera()
        }
    }

    private fun initializeCamera() {
        try {
            binding.cameraView.apply {
                visibility = SurfaceView.VISIBLE
                setCameraPermissionGranted()
                setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT)
                setCvCameraViewListener(this@MainActivity)
                enableView()
            }
            Log.i("CameraOpenCv", "Camera initialized successfully.")
        } catch (e: Exception) {
            Log.e("CameraOpenCv", "Failed to initialize camera: ${e.message}", e)
        }
    }

    private fun loadCascadeClassifier() {
        try {
            val inputStream = resources.openRawResource(R.raw.hand_cascade)
            val cascadeDir = getDir("cascade", MODE_PRIVATE)
            val cascadeFile = File(cascadeDir, "hand_cascade.xml")

            inputStream.use { input ->
                FileOutputStream(cascadeFile).use { output ->
                    input.copyTo(output)
                }
            }

            cascadeClassifier = CascadeClassifier(cascadeFile.absolutePath).apply {
                if (empty()) {
                    Log.e("CascadeClassifier", "Failed to load cascade")
                } else {
                    Log.i("CascadeClassifier", "Cascade loaded successfully")
                }
            }

            cascadeFile.delete()
            cascadeDir.delete()
        } catch (e: Exception) {
            Log.e("CascadeClassifier", "Error loading cascade: ${e.message}", e)
        }
    }

    private fun setupVolumeControl() {
        val maxVolume = audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
        val currentVolume = audioManager.getStreamVolume(AudioManager.STREAM_MUSIC)

        binding.volumeSeekbar.apply {
            max = maxVolume
            progress = currentVolume
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                @SuppressLint("SetTextI18n")
                override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                    audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, progress, 0)
                    binding.volumeText.text = "Volume Level: $progress"
                }

                override fun onStartTrackingTouch(seekBar: SeekBar?) {}
                override fun onStopTrackingTouch(seekBar: SeekBar?) {}
            })
        }
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        Log.i("CameraOpenCv", "Camera view started: $width x $height")
    }

    override fun onCameraViewStopped() {
        Log.i("CameraOpenCv", "Camera view stopped.")
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        val rgba = inputFrame?.rgba() ?: Mat()
        val gray = inputFrame?.gray() ?: Mat()

        val fingersCount = detectFingers(gray)
        runOnUiThread {
            adjustVolumeByFingers(fingersCount)
        }

        return rgba
    }

    private fun detectFingers(grayFrame: Mat): Int {
        if (grayFrame.empty() || grayFrame.type() != CvType.CV_8UC1) {
            Log.e("MatCheck", "Mat is not in the expected format")
            return 0
        }

        Imgproc.GaussianBlur(grayFrame, grayFrame, Size(5.0, 5.0), 0.0)
        val binaryMat = Mat()
        Imgproc.threshold(grayFrame, binaryMat, 60.0, 255.0, Imgproc.THRESH_BINARY_INV)

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(binaryMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        if (contours.isEmpty()) {
            Log.w("Contours", "No contours found")
            return 0
        }

        val largestContour = contours.maxByOrNull { Imgproc.contourArea(it) } ?: return 0

        if (Imgproc.contourArea(largestContour) < 500) {
            Log.w("Contours", "Largest contour too small")
            return 0
        }

        val hull = MatOfInt()
        Imgproc.convexHull(largestContour, hull, false)

        if (hull.empty()) {
            Log.w("ConvexHull", "Convex hull is empty")
            return 0
        }

        val defects = MatOfInt4()
        try {
            Imgproc.convexityDefects(largestContour, hull, defects)
        } catch (e: Exception) {
            Log.e("ConvexityDefects", "Error computing convexity defects: ${e.message}")
            return 0
        }

        var fingersCount = 0
        for (i in 0 until defects.rows()) {
            val defect = defects[i, 0]
            val startIdx = defect[0].toInt()
            val endIdx = defect[1].toInt()
            val farIdx = defect[2].toInt()

            val startPoint = largestContour[startIdx, 0]
            val endPoint = largestContour[endIdx, 0]
            val farPoint = largestContour[farIdx, 0]

            val a = distance(startPoint, endPoint)
            val b = distance(startPoint, farPoint)
            val c = distance(endPoint, farPoint)

            val angle = acos((b * b + c * c - a * a) / (2 * b * c))
            if (angle < Math.PI / 2) {
                fingersCount++
            }
        }

        return fingersCount.coerceAtMost(5)
    }

    private fun distance(point1: DoubleArray, point2: DoubleArray): Double {
        return sqrt(
            (point1[0] - point2[0]).pow(2.0) + (point1[1] - point2[1]).pow(2.0)
        )
    }

    @SuppressLint("SetTextI18n")
    private fun adjustVolumeByFingers(fingers: Int) {
        val maxVolume = audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
        val newVolume = (maxVolume * fingers.coerceIn(0, 5) / 5)

        audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, newVolume, 0)
        binding.volumeSeekbar.progress = newVolume
        binding.volumeText.text = "Volume: $newVolume"
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::binding.isInitialized) {
            binding.cameraView.disableView()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1 && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            initializeCamera()
        } else {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show()
        }
    }
}