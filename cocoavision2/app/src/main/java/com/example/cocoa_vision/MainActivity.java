package com.example.cocoa_vision;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import com.example.cocoa_vision.ml.Model;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    Button camera, gallery; // Sirve para declarar los botones
    ImageView imageView; // Sirve para declarar la imagen
    TextView result; // Sirve para declarar los textos
    int imageSize1 = 316; // Tamaño de la imagen
    int imageSize2 = 212; // Tamaño de la imagen

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        /*
         * Este método sirve para inicializar la actividad y se llama cuando se crea la actividad.
         */

        super.onCreate(savedInstanceState); // Crea la actividad y la muestra en la pantalla
        setContentView(R.layout.activity_main); // Carga el layout de la actividad en la pantalla

        camera = findViewById(R.id.button); // Asigna el botón de la cámara a la variable camera gracias al id del botón
        gallery = findViewById(R.id.button2); // Asigna el botón de la galería a la variable gallery gracias al id del botón2

        result = findViewById(R.id.result);  // Asigna el texto de la predicción a la variable result encontrándolo medinte al id del texto
        imageView = findViewById(R.id.imageView); // Asigna la imagen a la variable imageView gracias al id de la imagen

        camera.setOnClickListener(new View.OnClickListener() { // Asigna la acción de hacer click al botón de la cámara a la variable camera


            @Override
            public void onClick(View view) { // Este método se ejecuta cuando se hace click en el botón de la cámara
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) { // Si la versión de Android es mayor o igual a la versión 6 se ejecuta el siguiente código
                    if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {  // Si la aplicación tiene permiso para usar la cámara se ejecuta el siguiente código
                        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE); // Crea un intent para abrir la cámara y tomar una foto a la variable cameraIntent
                        startActivityForResult(cameraIntent, 3); // Inicia la actividad para tomar una foto y la asigna a la variable cameraIntent
                    } else { // Si la aplicación no tiene permiso para usar la cámara se ejecuta el siguiente código
                        requestPermissions(new String[]{Manifest.permission.CAMERA}, 100); // Pide permiso para usar la cámara
                    }
                }
            }
        });
        gallery.setOnClickListener(new View.OnClickListener() { // Asigna la acción de hacer click al botón de la galería a la variable gallery
            @Override
            public void onClick(View view) { // Este método se ejecuta cuando se hace click en el botón de la galería
                // Crea un intent para abrir la galería y seleccionar una foto a la variable cameraIntent
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                // Inicia la actividad para seleccionar una foto
                startActivityForResult(cameraIntent, 1);
            }
        });

    }
    public void classifyImage(Bitmap image){ // Este método sirve para clasificar la imagen
        try {
            Model model = Model.newInstance(getApplicationContext()); // Crea un modelo de la red neuronal y lo asigna a la variable model

            // Crea un tensor de entrada para la red neuronal y lo asigna a la variable inputFeature0
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 316, 212, 3}, DataType.FLOAT32);
            // Crea un buffer de bytes para la imagen y lo asigna a la variable byteBuffer
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize1 * imageSize2 * 3);
            // Asigna el orden de los bytes a la variable byteBuffer
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize1 * imageSize2]; // Crea un arreglo de enteros para la imagen y lo asigna a la variable intValues
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight()); // Asigna los pixeles de la imagen a la variable intValues
            int pixel = 0; // Crea una variable para los pixeles y se le asigna el valor de 0
            // Itera sobre cada pixel de la imagen y extrae los valores de R, G y B. Agrega esos valores individualmente al buffer de bytes
            for(int i = 0; i < imageSize1; i ++){
                for(int j = 0; j < imageSize2; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer); // Carga el buffer de bytes en el tensor de entrada

            // Ejecuta la inferencia del modelo y obtiene el resultado
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // Encuentra el índice de la clase con la mayor confianza.
            int maxPos = 0; // Crea una variable para la posición de la clase con mayor confianza y se le asigna el valor de 0
            float maxConfidence = 0; // Crea una variable para la confianza de la clase con mayor confianza y se le asigna el valor de 0
            for (int i = 0; i < confidences.length; i++) { // Itera sobre cada clase y encuentra la clase con mayor confianza
                // Si la confianza de la clase es mayor a la confianza de la clase con mayor confianza Se asigna la confianza de la clase a la variable maxConfidence
                // y se asigna el índice de la clase a la variable maxPos
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"Monilia", "Sana" }; // Crea un arreglo con las clases y se le asigna el nombre de las clases
            result.setText(classes[maxPos]); // Asigna el nombre de la clase con mayor confianza a la variable result

            // Libera los recursos del modelo si ya no se usan
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) { // Este método se ejecuta cuando se regresa de la actividad de la cámara o la galería
        if(resultCode == RESULT_OK){ // Si el resultado de la actividad es correcto se ejecuta el siguiente código

            if(requestCode == 3){ // Si la actividad es la de la cámara se ejecuta el siguiente código
                // Obtiene la imagen de la actividad de la cámara y la asigna a la variable image
                Bitmap image = (Bitmap) data.getExtras().get("data");
                // Crea una variable para la dimensión de la imagen y se le asigna el valor de la dimensión más pequeña de la imagen
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension); // Extrae la imagen
                imageView.setImageBitmap(image);   // setImageBitmap() asigna la imagen a la variable imageView

                image = Bitmap.createScaledBitmap(image, imageSize1, imageSize2, false); // Cambia el tamaño de la imagen a 416x312
                classifyImage(image); // Clasifica la imagen
            }else{
                Uri dat = data.getData(); // Obtiene la ruta de la imagen seleccionada y la asigna a la variable dat
                Bitmap image = null; // Crea una variable para la imagen y se le asigna el valor de null
                try {
                    // Obtiene la imagen de la galería y la asigna a la variable image
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace(); // Imprime el error en la consola
                }
                imageView.setImageBitmap(image); // setImageBitmap() asigna la imagen a la variable imageView

                // Crea una imagen con las dimensiones de la imagen de entrada de la red neuronal y la asigna a la variable image
                image = Bitmap.createScaledBitmap(image, imageSize1, imageSize2, false);
                classifyImage(image); // Clasifica la imagen
            }
        }
        super.onActivityResult(requestCode, resultCode, data); // Ejecuta el método onActivityResult de la clase padre
    }
}
