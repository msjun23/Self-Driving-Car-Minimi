# Self-Driving-Car-Minimi

**Personal project at Baram, Academic group** </br>
**Division of Robotics, Kwangwoon University at Seoul, Korea** </br>
**Term:** 09.2020. ~ 11.2020. </br>
**Intro:** Self Driving Car project using OpenCV and Tensorflow API

**Hardware:** Jetson Nano(for **vision, learning**), NUCLEO64-STM32F401RE(for **motor control**) </br>
**OS:** Jetson Nano -> Ubuntu / F401RE -> Window </br>
**Programming Language:** Python & C

---

Basically detecting & following lane. And using Tensorflow API, detect human(or car, but not in result video...) for emergency stop.

Used Jetson Nano for image processing and learning. Result of lane detecting and human detecting is transmitted to embedded board, F401RE for motor control.

For motor control, used PID controller. Error input is deviation, from center of lane and robot's current position(center of horizontal camera image).

![slide1](/result/image/slide1.PNG) | ![slide2](/result/image/slide2.PNG)
---|---|
![slide3](/result/image/slide3.PNG) | ![slide4](/result/image/slide4.PNG)


<img src="/result/output2.gif" width="50%" height="50%">
