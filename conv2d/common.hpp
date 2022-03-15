void display_value(float * outbuf, int size){
    std::cout << outbuf[rand()%size] << "\n";
}

void display_time(std::chrono::time_point<std::chrono::system_clock> t1,  std::chrono::time_point<std::chrono::system_clock>t2, int size,
        int nbthreads, std::string compiler, std::string option, std::string name){

    std::chrono::duration< double > fs = t2 - t1;
    std::chrono::microseconds d = std::chrono::duration_cast< std::chrono::microseconds >( fs );
    //std::cout << fs.count() << "s\n";
    //std::cerr << nbthreads << ";" ;
    std::cerr << compiler << ";" ;
    std::cerr << option << ";" ;
    std::cerr << name  << ";" ;
    std::cerr << d.count() /1000.<< "\n";
}

