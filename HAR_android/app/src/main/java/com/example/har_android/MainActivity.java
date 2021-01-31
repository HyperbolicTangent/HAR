package com.example.har_android;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;
import java.io.IOException;
import android.speech.tts.TextToSpeech;
import androidx.appcompat.app.AppCompatActivity;
import android.widget.TextView;
import android.widget.TableRow;
//import com.example.har_android.ml.Model2;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity implements SensorEventListener, TextToSpeech.OnInitListener {

    private static final int N_SAMPLES = 250;
    private static final int samplerate = 20000;
    private static int prevIdx = -1;
    private static List<Float> ax; private static List<Float> ax_norm_window;
    private static List<Float> ay; private static List<Float> ay_norm_window;
    private static List<Float> az; private static List<Float> az_norm_window;
    private static List<Float> gx; private static List<Float> gx_norm_window;
    private static List<Float> gy; private static List<Float> gy_norm_window;
    private static List<Float> gz; private static List<Float> gz_norm_window;

    private SensorManager mSensorManager;
    private Sensor mAccelerometer;
    private Sensor mGyroscope;

    private float[][] results;
    private HAR_classifier classifier;
    private TextToSpeech textToSpeech;

    private TextView walkingTextView;
    private TextView walkingupstairsTextView;
    private TextView walkingdownstairsTextView;
    private TextView sittingTextView;
    private TextView standingTextView;
    private TextView layingTextView;
    private TextView standtositTextView;
    private TextView sittostandTextView;
    private TextView sittolieTextView;
    private TextView lietositTextView;
    private TextView standtolieTextView;
    private TextView lietostandTextView;

    /*private TableRow walkingTableRow;
    private TableRow walkingupstairsTableRow;
    private TableRow walkingdownstairsTableRow;
    private TableRow sittingTableRow;
    private TableRow standingTableRow;
    private TableRow layingTableRow;
    private TableRow standtositTableRow;
    private TableRow sittostandTableRow;
    private TableRow sittolieTableRow;
    private TableRow lietositTableRow;
    private TableRow standtolieTableRow;
    private TableRow lietostandTableRow;*/



    private String[] labels = {"walking", "walkingupstairs", "walkingdownstairs", "sitting", "standing", "laying", "standtosit",
            "sittostand", "sittolie", "lietosit", "standtolie", "lietostand"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ax = new ArrayList<>();
        ax_norm_window = new ArrayList<>();
        ay = new ArrayList<>();
        ay_norm_window = new ArrayList<>();
        az = new ArrayList<>();
        az_norm_window = new ArrayList<>();
        gx = new ArrayList<>();
        gx_norm_window = new ArrayList<>();
        gy = new ArrayList<>();
        gy_norm_window = new ArrayList<>();
        gz = new ArrayList<>();
        gz_norm_window = new ArrayList<>();

        walkingTextView = (TextView) findViewById(R.id.walking_prob);
        walkingupstairsTextView = (TextView) findViewById(R.id.walkingupstairs_prob);
        walkingdownstairsTextView = (TextView) findViewById(R.id.walkingdownstairs_prob);
        sittingTextView = (TextView) findViewById(R.id.sitting_prob);
        standingTextView = (TextView) findViewById(R.id.standing_prob);
        layingTextView = (TextView) findViewById(R.id.laying_prob);
        standtositTextView = (TextView) findViewById(R.id.standtosit_prob);
        sittostandTextView = (TextView) findViewById(R.id.sittostand_prob);
        sittolieTextView = (TextView) findViewById(R.id.sittolie_prob);
        lietositTextView = (TextView) findViewById(R.id.lietosit_prob);
        standtolieTextView = (TextView) findViewById(R.id.standtolie_prob);
        lietostandTextView = (TextView) findViewById(R.id.lietostand_prob);

        /*walkingTableRow = (TableRow) findViewById(R.id.walking_row);
        walkingupstairsTableRow = (TableRow) findViewById(R.id.walkingupstairs_row);
        walkingdownstairsTableRow = (TableRow) findViewById(R.id.walkingdownstairs_row);
        sittingTableRow = (TableRow) findViewById(R.id.sitting_row);
        standingTableRow = (TableRow) findViewById(R.id.standing_row);
        layingTableRow = (TableRow) findViewById(R.id.laying_row);
        standtositTableRow = (TableRow) findViewById(R.id.standtosit_row);
        sittostandTableRow = (TableRow) findViewById(R.id.sittostand_row);
        sittolieTableRow = (TableRow) findViewById(R.id.sittolie_row);
        lietositTableRow = (TableRow) findViewById(R.id.lietosit_row);
        standtolieTableRow = (TableRow) findViewById(R.id.standtolie_row);
        lietostandTableRow = (TableRow) findViewById(R.id.lietostand_row);*/


        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);

        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mSensorManager.registerListener(this, mAccelerometer, samplerate);

        mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        mSensorManager.registerListener(this, mGyroscope, samplerate);

        try {
            classifier = new HAR_classifier(getApplicationContext()); //try/catch?
        } catch (IOException e) {
            Log.e("tfliteSupport", "Error reading model", e);
        }

        //classifier = new Model2(getApplicationContext());
        /*try {
            Model2 model = Model2.newInstance(getApplicationContext());
        } catch (IOException e){
            Log.e("tfliteSupport", "Error reading model", e);
        }*/

        textToSpeech = new TextToSpeech(this, this);
        textToSpeech.setLanguage(Locale.US);
    }

    @Override
    public void onInit(int status) {
        Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                if (results == null || results.length == 0) {
                    return;
                }
                float max = -1;
                int idx = -1;
                for (int i = 0; i < results.length; i++) {
                    if (results[0][i] > max) {
                        idx = i;
                        max = results[0][i];
                    }
                }
                if(max > 0.50 && idx != prevIdx) {
                    textToSpeech.speak(labels[idx], TextToSpeech.QUEUE_ADD, null,
                            Integer.toString(new Random().nextInt()));
                    prevIdx = idx;
                }
            }
        }, 1000, 5000);
    }

    @Override
    protected void onPause() {
        getSensorManager().unregisterListener(this);
        super.onPause();
    }

    protected void onResume() {
        super.onResume();
        getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_ACCELEROMETER), samplerate);
        getSensorManager().registerListener(this, getSensorManager().getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION), samplerate);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        activityPrediction();
        Sensor sensor = event.sensor;
        if (sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            ax.add(event.values[0]);
            ay.add(event.values[1]);
            az.add(event.values[2]);

        } else if (sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            gx.add(event.values[0]);
            gy.add(event.values[1]);
            gz.add(event.values[2]);

        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }

    private void activityPrediction() {
        List<Float> data = new ArrayList<>();
        if (ax.size() == N_SAMPLES && ay.size() == N_SAMPLES && az.size() == N_SAMPLES
                && gx.size() == N_SAMPLES && gy.size() == N_SAMPLES && gz.size() == N_SAMPLES) {
            float ax_sum = 0; float ay_sum = 0; float az_sum = 0;
            float gx_sum = 0; float gy_sum = 0; float gz_sum = 0;
            //calculate z score norm
            for (int i = 0; i < N_SAMPLES; i++){
                ax_sum =  ax_sum + ax.get(i); ay_sum = ay_sum + ay.get(i); az_sum = az_sum + az.get(i);
                gx_sum = gx_sum + gx.get(i); gy_sum = gy_sum + gy.get(i); gz_sum = gz_sum + gz.get(i);
            }
            float ax_mean = ax_sum/N_SAMPLES; float ay_mean = ay_sum/N_SAMPLES; float az_mean = az_sum/N_SAMPLES;
            float gx_mean = gx_sum/N_SAMPLES; float gy_mean = gy_sum/N_SAMPLES; float gz_mean = gz_sum/N_SAMPLES;
            float ax_var = 0; float ay_var = 0; float az_var = 0; float gx_var = 0; float gy_var = 0; float gz_var = 0;
            for (int i = 0; i < N_SAMPLES; i++){
                float ax_s = ax.get(i) - ax_mean; ax_var += Math.pow(ax_s, 2);
                float ay_s = ay.get(i) - ay_mean; ay_var += Math.pow(ay_s, 2);
                float az_s = az.get(i) - az_mean; az_var += Math.pow(az_s, 2);
                float gx_s = gx.get(i) - gx_mean; gx_var += Math.pow(gx_s, 2);
                float gy_s = ax.get(i) - gy_mean; gy_var += Math.pow(gy_s, 2);
                float gz_s = gz.get(i) - gz_mean; gz_var += Math.pow(gz_s, 2);
            }
            float ax_StdDev = (float)Math.sqrt(ax_var/N_SAMPLES); float ay_StdDev = (float)Math.sqrt(ay_var/N_SAMPLES); float az_StdDev = (float)Math.sqrt(az_var/N_SAMPLES);
            float gx_StdDev = (float)Math.sqrt(gx_var/N_SAMPLES); float gy_StdDev = (float)Math.sqrt(gy_var/N_SAMPLES); float gz_StdDev = (float)Math.sqrt(gz_var/N_SAMPLES);

            /*List<Float> data = new ArrayList<>();
            data.addAll(ax);
            data.addAll(ay);
            data.addAll(az);
            data.addAll(gx);
            data.addAll(gy);
            data.addAll(gz);*/

            float[][][] input_3d = new float[1][250][6];
            for (int n = 0; n < N_SAMPLES; n++) {
                input_3d[0][n][0] = (ax.get(n) - ax_mean)/ax_StdDev;
                input_3d[0][n][1] = (ay.get(n) - ay_mean)/ay_StdDev;
                input_3d[0][n][2] = (az.get(n) - az_mean)/az_StdDev;
                input_3d[0][n][3] = (gx.get(n) - gx_mean)/gx_StdDev;
                input_3d[0][n][4] = (gy.get(n) - gy_mean)/gy_StdDev;
                input_3d[0][n][5] = (gz.get(n) - gz_mean)/gz_StdDev;
            }


            /*float ax_norm, ay_norm, az_norm, gx_norm, gy_norm, gz_norm;
            for(int n = 0; n < N_SAMPLES; n++){
                ax_norm = (ax.get(n) - ax_mean)/ax_StdDev; ay_norm = (ay.get(n) - ay_mean)/ay_StdDev; az_norm = (az.get(n) - az_mean)/az_StdDev;
                gx_norm = (gx.get(n) - gx_mean)/gx_StdDev; gy_norm = (gy.get(n) - gy_mean)/gy_StdDev; gz_norm = (gz.get(n) - gz_mean)/gz_StdDev;
                ax_norm_window.add(ax_norm); ay_norm_window.add(ay_norm); az_norm_window.add(az_norm);
                gx_norm_window.add(gx_norm); gy_norm_window.add(gy_norm); gz_norm_window.add(gz_norm);
            }
            data.addAll(ax_norm_window.subList(0, N_SAMPLES));
            data.addAll(ay_norm_window.subList(0, N_SAMPLES));
            data.addAll(az_norm_window.subList(0, N_SAMPLES));
            data.addAll(gx_norm_window.subList(0, N_SAMPLES));
            data.addAll(gy_norm_window.subList(0, N_SAMPLES));
            data.addAll(gz_norm_window.subList(0, N_SAMPLES));*/

            results = classifier.predictions(input_3d);

            /*try {
                Model2 model = Model2.newInstance(getApplicationContext());

                // Creates inputs for reference.
                TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 250, 6},  DataType.FLOAT32);
                inputFeature0.loadBuffer(input_3d);

                // Runs model inference and gets result.
                Model2.Outputs outputs = model.process(inputFeature0);
                TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                // Releases model resources if no longer used.
                model.close();
            } catch (IOException e) {
                // TODO Handle the exception
            }*/


            walkingTextView.setText(Float.toString(round(results[0][0], 2)));
            walkingupstairsTextView.setText(Float.toString(round(results[0][1], 2)));
            walkingdownstairsTextView.setText(Float.toString(round(results[0][2], 2)));
            sittingTextView.setText(Float.toString(round(results[0][3], 2)));
            standingTextView.setText(Float.toString(round(results[0][4], 2)));
            layingTextView.setText(Float.toString(round(results[0][5], 2)));
            standtositTextView.setText(Float.toString(round(results[0][6], 2)));
            sittostandTextView.setText(Float.toString(round(results[0][7], 2)));
            sittolieTextView.setText(Float.toString(round(results[0][8], 2)));
            lietositTextView.setText(Float.toString(round(results[0][9], 2)));
            standtolieTextView.setText(Float.toString(round(results[0][10], 2)));
            lietostandTextView.setText(Float.toString(round(results[0][11], 2)));

            ax.clear(); gx.clear();
            ay.clear(); gy.clear();
            az.clear(); gz.clear();
            ax_norm_window.clear(); ay_norm_window.clear(); az_norm_window.clear();
            gx_norm_window.clear(); gy_norm_window.clear(); gz_norm_window.clear();

        }
    }


    private float[] toFloatArray(List<Float> list) {
        int i = 0;
        float[] array = new float[list.size()];

        for (Float f : list) {
            array[i++] = (f != null ? f : Float.NaN);
        }
        return array;
    }

    private static float round(float d, int decimalPlace) {
        BigDecimal bd = new BigDecimal(Float.toString(d));
        bd = bd.setScale(decimalPlace, BigDecimal.ROUND_HALF_UP);
        return bd.floatValue();
    }

    private SensorManager getSensorManager() {
        return (SensorManager) getSystemService(SENSOR_SERVICE);
    }

}