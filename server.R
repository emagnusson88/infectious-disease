library(shiny)
library(coronavirus)
library(ggplot2)
library(dplyr)
library(readr)
library(reticulate)

# Get daily updates from GitHub
#devtools::install_github("RamiKrispin/coronavirus",force=TRUE)
#update_dataset()

covid <- coronavirus

function(input,output){
  
  output$ui <- renderUI({
    if (is.null(input$dataSource))
      return()
    
    switch(input$dataSource,
           "Upload Data" = fileInput("dataFile","Choose CSV File",accept = ".csv")
    )
  })
  
  reticulate::source_python('python_test.py')
  
  output$message <- renderText({
    return(python_test())
  })
  
  c <- eventReactive(input$update,{
    covid %>%
      filter(country == input$countryInput,
             type == "confirmed",
             date >= '2021-06-01',
             date <= '2021-10-01')
  })
  
  output$covidplot <- renderPlot({
    ggplot(c(), aes(x=date,y=cases,color=country)) +
      geom_point() +
      geom_line() +
      xlab("Date") +
      ylab("Cases") +
      ggtitle("Cases Over Time")
  })
  
}
